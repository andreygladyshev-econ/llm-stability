"""Облачный раннер (OpenRouter): тот же запрос, та же схема, тот же подсчёт кодом — другая модель.

ЗАЩИТА ДЕНЕГ — три независимых слоя:
  1. Лимит на самом ключе в OpenRouter ($20). Последний рубеж, от нашего кода не зависит.
  2. Раннер НИКОГДА не отправляет запрос, который в худшем случае (весь max_tokens по цене выхода) может вывести
     за общий стоп ($18 по умолчанию) или за бюджет модели. Потрачено считается по ФАКТИЧЕСКОЙ цене каждого
     ответа (usage.cost) и каждые 10 запросов сверяется с OpenRouter (/api/v1/key).
  3. Перед моделью — прогноз по ценам каталога; после первого запроса — прогноз по факту. Если прогноз выше
     бюджета модели — модель пропускается целиком, а не «пока не кончатся деньги».
  Плюс: потолок max_tokens на ответ, размышление минимальное, три провала подряд — модель снимается,
  аварийный стоп — создать файл logs/cloud.STOP (раннер остановится перед следующим запросом).

Провайдер закрепляется: первый ответ показывает, кто отвечал, дальше все запросы модели идут только к нему
(allow_fallbacks = false). Точность весов провайдера записывается в каждый ответ.
Отличия от локального запроса (записаны в meta): нет min_p, top_k и repeat_penalty — их поддерживают не все
провайдеры, а require_parameters иначе отсёк бы половину.

Запуск: .venv/bin/python src/cloud.py plan  queues/cloud.toml   прогноз денег, ничего не отправляет
        .venv/bin/python src/cloud.py run   queues/cloud.toml --only <модель>   одна модель (так и работаем:
                                                            после каждой — разбор денег и качества, потом следующая)
        .venv/bin/python src/cloud.py chain queues/cloud.toml   все модели по очереди; после каждой пауза на разбор:
                                                            дальше — `touch logs/cloud.GO`, стоп — `touch logs/cloud.STOP`
        .venv/bin/python src/cloud.py selftest
"""
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import night  # noqa: E402  — очередь, имена прогонов, сборка запроса, хеши спеки и текста
import texts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW, LOGS = ROOT / "raw", ROOT / "logs"
LEDGER, STOP = LOGS / "cloud_ledger.jsonl", LOGS / "cloud.STOP"
API = "https://openrouter.ai/api/v1"
GLOBAL_CAP = 18.0          # общий стоп, $ — ниже лимита ключа ($20)
MAX_TOKENS = 9000          # потолок ответа без размышления: наш ответ ~3,4 тыс. токенов
MAX_TOKENS_REASONING = 16000  # у моделей со встроенным размышлением оно входит в тот же потолок (документация OpenRouter)
OUT_EST = 3400             # ожидаемый ответ для прогноза
REASONING_OUT_FACTOR = 1.8  # у «думающих» моделей ответ длиннее
CHARS_PER_TOKEN = 3.2       # грубо для русского текста; уточняется по первому ответу


def key():
    return (Path.home() / ".openrouter_key").read_text().strip()


def get(path):
    return httpx.get(f"{API}{path}", headers={"Authorization": f"Bearer {key()}"}, timeout=60).json()


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def log(*a):
    print(now(), *a, flush=True)


# ---------- деньги ----------

def spent():
    """Сколько потрачено этим раннером по журналу: всего и по моделям."""
    total, by = 0.0, {}
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            r = json.loads(line)
            total += r["cost"]
            by[r["model"]] = by.get(r["model"], 0.0) + r["cost"]
    return total, by


MAX_TOKENS_THINK = 32000   # опыт «думает ли модель лучше»: размышление включено на заданном уровне


def reasoning_setup(prices, job=None):
    """Как у локальной модели — без размышления. Где оно встроено и не выключается — минимальный уровень.

    22.09: Nemotron Nano с effort=low потратила все 9000 токенов на размышление и вернула пустой ответ;
    проба «2+2» показала, у каких моделей размышление выключается (logs/cloud_reasoning_probe_22.09.txt).
    → (параметр reasoning или None, потолок ответа, метка для meta)"""
    if job and job.get("reasoning_effort"):                 # 23.09: опыт с включённым размышлением
        e = job["reasoning_effort"]
        return {"effort": e, "exclude": True}, job.get("max_tokens", MAX_TOKENS_THINK), f"включено, {e}"
    r = prices.get("reasoning") or {}
    if "reasoning" not in prices.get("params", ()):
        return None, MAX_TOKENS, "нет"
    if not r.get("mandatory"):
        return {"enabled": False}, MAX_TOKENS, "выключено"
    effort = (r.get("supported_efforts") or ["low"])[-1]       # уровни идут от высокого к низкому
    return {"effort": effort, "exclude": True}, MAX_TOKENS_REASONING, f"встроено, {effort}"


def worst_case(prices, in_tokens, max_tokens=None, job=None):
    """Самая дорогая цена одного запроса: весь потолок ответа по цене выхода."""
    max_tokens = max_tokens or reasoning_setup(prices, job)[1]
    return in_tokens * prices["in"] + max_tokens * prices["out"]


def may_send(total, model_spent, budget, wc, cap=GLOBAL_CAP):
    """Можно ли отправить запрос, чтобы даже в худшем случае не выйти ни за общий стоп, ни за бюджет модели."""
    if total + wc > cap:
        return False, f"общий стоп: потрачено ${total:.2f}, худший запрос ${wc:.3f}, стоп ${cap}"
    if model_spent + wc > budget:
        return False, f"бюджет модели: потрачено ${model_spent:.2f}, худший запрос ${wc:.3f}, бюджет ${budget}"
    return True, ""


# ---------- каталог ----------

def catalog():
    cache = LOGS / "cloud_catalog.json"
    try:
        d = httpx.get(f"{API}/models", timeout=60).json()["data"]
        cache.write_text(json.dumps(d, ensure_ascii=False))
    except Exception:
        d = json.loads(cache.read_text())
    out = {}
    for m in d:
        p = m.get("pricing") or {}
        out[m["id"]] = {"in": float(p.get("prompt") or 0), "out": float(p.get("completion") or 0),
                        "params": set(m.get("supported_parameters") or []), "ctx": m.get("context_length"),
                        "open": bool(m.get("hugging_face_id")),
                        "reasoning": m.get("reasoning") or {}}
    return out


QUANT_RANK = {"bf16": 0, "fp16": 0, "fp32": 0, "fp8": 1, "mxfp8": 1, "int8": 2, "fp6": 3, "unknown": 4, None: 4,
              "fp4": 5, "mxfp4": 5, "nvfp4": 5, "int4": 5}


def pick_providers(model, need, price_out=None, soft=("structured_outputs",)):
    """Провайдеры по порядку: точнее веса → быстрее (доступность ≥ 95% — фильтр). Только те, кто знает все параметры запроса.

    22.09: первый запрос Qwen 27B без выбора попал на перегруженного провайдера (429) — запасных нет, модель стоит.
    → список меток (tag, напр. «deepinfra/bf16») и описание для лога."""
    try:
        eps = get(f"/models/{model}/endpoints")["data"]["endpoints"]
    except Exception:
        return [], "список провайдеров недоступен — выбор OpenRouter"
    TIERS = {"fast", "flex", "priority"}
    ref = price_out or min((float(e["pricing"].get("completion") or 0) for e in eps), default=0)

    def tps_of(e):
        return (e.get("throughput_last_30m") or {}).get("p50") or 0

    # Требуем только то, что хоть кто-то умеет: у Opus 5 температуру не принимает НИ ОДИН провайдер, хотя
    # каталог модели говорит обратное. Невозможные параметры снимаются из запроса (пишется в ответ), а
    # возможные — остаются обязательными (у Nemotron Super строгую схему держит только один провайдер).
    need = {k for k in need if any(k in set(e.get("supported_parameters") or []) for e in eps)}

    def fits(e, min_tps, max_price, min_up=97):
        return (need <= set(e.get("supported_parameters") or []) and e.get("status", 0) == 0
                and (e.get("uptime_last_30m") or 0) >= min_up and not TIERS & set(e["tag"].split("/")[1:])
                and ((e.get("latency_last_30m") or {}).get("p50") or 0) <= 20000
                and float(e["pricing"].get("completion") or 0) <= ref * max_price and tps_of(e) >= min_tps)
    # Разумная экономия: не переплачивать (≤ 1,25 цены каталога) и не ждать (≥ 25 ток/с). Послабления по
    # очереди: цена, затем доступность, затем скорость.
    def ladder():
        return ([e for e in eps if fits(e, 25, 1.25)] or [e for e in eps if fits(e, 25, 1.6)]
                or [e for e in eps if fits(e, 25, 1.6, 95)] or [e for e in eps if fits(e, 10, 2, 95)]
                or [e for e in eps if fits(e, 0, 3, 90)])
    ok = ladder()
    if not ok and set(soft) & need:
        # строгая схема желательна, но не любой ценой: у Nemotron Super её держит только аварийный провайдер,
        # у Opus 5 схема и температура есть только у РАЗНЫХ провайдеров. Тогда просим JSON без проверки схемы —
        # модель формирует его сама, а ранний стоп снимет модель, если ответы не разбираются.
        need = need - set(soft)
        ok = ladder()
    # Точность весов важнее скорости, но не любой ценой: провайдер медленнее лучшего в 2,5 раза не берётся
    # (у Gemma все bf16 дают 5 ток/с — это 9 часов на модель, берём fp8/fp4 на 30 ток/с).
    best = max((tps_of(e) for e in ok), default=0)
    ok = [e for e in ok if tps_of(e) * 2.5 >= best] or ok
    tps = [(e.get("throughput_last_30m") or {}).get("p50") or 0 for e in ok]
    # общее подмножество по тем, кого реально пошлём (первая тройка) — иначе теряем seed из-за дальнего запасного
    supported = set.intersection(*(set(e.get("supported_parameters") or []) for e in ok[:3])) if ok else None
    return ([e["tag"] for e in ok],
            ", ".join(f"{e['tag']} ({t} ток/с)" for e, t in list(zip(ok, tps))[:4]),
            tps[0] if tps else 0, supported)


def endpoint_quant(model, provider):
    """Точность весов у закреплённого провайдера — из списка его эндпоинтов."""
    try:
        eps = get(f"/models/{model}/endpoints")["data"]["endpoints"]
        return next((e.get("quantization") for e in eps if e.get("provider_name") == provider), None)
    except Exception:
        return None


# ---------- запрос ----------

def cloud_body(job, report_text, run, cat, provider=None, supported=None):
    body, request_hash = night.build_request(job, report_text, run)
    for k in ("top_k", "repeat_penalty", "min_p"):
        body.pop(k, None)
    if body["messages"][-1]["role"] == "assistant":          # подстановка пустого <think> — только для локальных
        body["messages"].pop()
    reasoning, body["max_tokens"], _ = reasoning_setup(cat[job["model"]], job)
    ok_params = supported if supported is not None else cat[job["model"]]["params"]
    if job.get("seed") is not None and "seed" in ok_params:   # MiMo и Opus seed не знают:
        body["seed"] = job["seed"]                  # с require_parameters запрос иначе не ушёл бы никуда
    if reasoning:
        body["reasoning"] = reasoning
    body.pop("stream", None)                          # без потока — по умолчанию
    dropped = [k for k in ("temperature", "top_p", "seed") if k in body and k not in ok_params]
    for k in dropped:                                 # GPT-5.6 Sol, GPT-6 Astra — без температуры; Opus — без top_p.
        body.pop(k, None)                             # с require_parameters иначе запрос не уходит никуда
    if (supported is not None and "structured_outputs" not in supported
            and (body.get("response_format") or {}).get("type") == "json_schema"):
        # провайдер не умеет строгую схему: с require_parameters запрос иначе не уходит никуда (404).
        # Просим обычный JSON — модель формирует структуру сама; в ответе это помечено.
        body["response_format"] = {"type": "json_object"}
        dropped = dropped + ["json_schema"]
    body["_dropped"] = dropped
    # Метка эндпоинта вида «deepinfra/bf16» в маршрутизации не принимается (404 «нет подходящих эндпоинтов»):
    # имя провайдера и точность весов передаются раздельно — slug + provider.quantizations.
    prov = [provider] if isinstance(provider, str) else list(provider or [])
    slugs, quants = [], []
    for tag in prov:
        head, _, tail = tag.partition("/")
        if tail in QUANT_RANK:
            slugs.append(head); quants.append(tail)
        else:
            slugs.append(tag)
    body["provider"] = {"allow_fallbacks": False, "require_parameters": True,
                        **({"order": slugs, "only": slugs} if slugs else {}),
                        **({"quantizations": sorted(set(quants))} if quants and len(set(quants)) == 1 else {})}
    body["usage"] = {"include": True}
    return body, request_hash


def pointless(t, cat):
    """Серия с температурой > 0 у модели, которая температуру не принимает, — копия боевого режима: не тратим."""
    return t["job"].get("temperature", 0) > 0 and "temperature" not in cat[t["job"]["model"]]["params"]


class NetFail:
    """Ответ-заглушка, когда сеть не ответила вовсе (обрыв, таймаут): считается провалом, денег не стоит."""
    status_code, text = 0, "сеть недоступна"

    def json(self):
        return {"error": {"message": self.text}}


def patience(times, expect=None):
    """Сколько ждать ответ: пока ответов мало — втрое дольше расчётного времени, дальше — вчетверо дольше обычного.

    Расчёт: ожидаемая длина ответа ÷ скорость провайдера (данные OpenRouter) + минута на очередь.
    Границы 3–15 мин. 22.09: провайдер зависал на запросе при обычных 40 с — одинаковые 15 минут для всех моделей
    и бессмысленны, и дороги. Повтор после обрыва посылает тот же запрос заново, клетка не теряется."""
    import statistics
    t = 3 * (expect or 300) if len(times) < 3 else 4 * statistics.median(times)
    return max(180, min(900, t))


def filled(content):
    """Доля показателей, где хоть одно поле заполнено. Gemini 3.8 Flash вернула 55 формально верных, но ПУСТЫХ
    ответов ($0,55 впустую): схема не помечает поля обязательными, и пустой объект ей соответствует."""
    i, j = content.find("["), content.find("{")     # список (Opus) или словарь — берём самый внешний
    try:
        if 0 <= i < (j if j >= 0 else len(content)):
            d = json.loads(content[i:content.rindex("]") + 1])
        else:
            d = json.loads(content[j:content.rindex("}") + 1])
    except (ValueError, TypeError):
        return 0.0
    if isinstance(d, dict):
        lst = next((v for v in d.values() if isinstance(v, list) and any(isinstance(x, dict) for x in v)), None)
        its = lst if lst is not None else list(d.values())
    else:
        its = d
    its = [x for x in its if isinstance(x, dict)]
    if not its:
        return 0.0
    return sum(any(str(v).strip() for k, v in x.items() if k != "id") for x in its) / len(its)


def send(body, timeout=600):
    r = NetFail()
    for attempt in range(4):
        try:
            r = httpx.post(f"{API}/chat/completions", json=body, timeout=timeout,
                           headers={"Authorization": f"Bearer {key()}", "X-OpenRouter-Cache": "false"})
        except httpx.HTTPError as e:              # обрыв Wi-Fi, таймаут, DNS — ждём и повторяем
            r = NetFail(); r.text = f"сеть: {type(e).__name__}"
            time.sleep(30 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            wait = r.headers.get("Retry-After", "")
            time.sleep(min(int(wait), 120) if wait.isdigit() else 10 * (attempt + 1))
            continue
        return r
    return r


# ---------- план и прогон ----------

def tasks_by_model(queue):
    _, tasks = night.load_queue(queue)
    pend = [t for t in night.pending(tasks)]
    order = night.model_order(pend)
    return {m: [t for t in pend if t["job"]["model"] == m] for m in order}


def in_tokens_est(t):
    text = (texts.text_dir(t["job"].get("text", "v1")) / f"{t['report']}.txt").read_text()
    spec, prompt = night.spec_bundle(t["job"]["spec"])
    return int((len(text) + len(spec) + len(prompt)) / CHARS_PER_TOKEN)


def projection(model, ts, cat):
    p = cat[model]
    base = REASONING_OUT_FACTOR if "reasoning" in p["params"] else 1.0

    def factor(t):          # размышление включено: ответ втрое-вчетверо длиннее
        return 3.5 if t["job"].get("reasoning_effort") else base
    return sum(in_tokens_est(t) * p["in"] + OUT_EST * factor(t) * p["out"] for t in ts if not pointless(t, cat))


def plan(queue):
    cat = catalog()
    total_so_far, by = spent()
    grand = 0.0
    print(f"{'модель':42} {'запросов':>8} {'прогноз':>9} {'бюджет':>8} {'худший запрос':>14}")
    for model, ts in tasks_by_model(queue).items():
        if model not in cat:
            print(f"{model:42} НЕТ В КАТАЛОГЕ — будет пропущена"); continue
        pr, budget = projection(model, ts, cat), ts[0]["job"].get("budget", 1.0)
        wc = worst_case(cat[model], in_tokens_est(ts[0]), job=ts[0]["job"])
        flag = "" if pr <= budget else "  ← прогноз выше бюджета, модель будет пропущена"
        grand += min(pr, budget)
        print(f"{model:42} {len(ts):>8} ${pr:>8.2f} ${budget:>7.2f} ${wc:>13.3f}{flag}")
    print(f"\nуже потрачено этим раннером: ${total_so_far:.2f}; прогноз остального: ${grand:.2f}; общий стоп: ${GLOBAL_CAP}")
    try:
        c = get("/credits")["data"]
        print(f"баланс аккаунта: свободно ${c['total_credits'] - c['total_usage']:.2f}")
    except Exception:
        pass


def run(queue, only=None):
    cat = catalog()
    c = get("/credits")["data"]
    free = c["total_credits"] - c["total_usage"]
    total0, _ = spent()          # баланс аккаунта уже учёл всё, что потрачено до этого запуска
    log(f"баланс аккаунта: свободно ${free:.2f}")
    requests_done = 0
    for model, ts in tasks_by_model(queue).items():
        if only and model != only:
            continue
        if model not in cat:
            log(f"пропуск {model}: нет в каталоге"); continue
        budget = ts[0]["job"].get("budget", 1.0)
        pr = projection(model, ts, cat)
        if pr > budget:
            log(f"пропуск {model}: прогноз ${pr:.2f} выше бюджета ${budget}"); continue
        if pr > GLOBAL_CAP - spent()[0]:
            log(f"пропуск {model}: прогноз ${pr:.2f} не влезает в остаток общего стопа ${GLOBAL_CAP - spent()[0]:.2f}"); continue
        provider, quant, fails, first_cost = None, None, 0, None
        b0, _ = cloud_body(ts[0]["job"], "x", ts[0]["run"], cat); b0.pop("_dropped")
        # что просим у провайдера; seed — необязательный: он усиливает проверку повторяемости, но из-за него
        # у MiMo оставался провайдер на 14 ток/с вместо 47. Если провайдер seed не знает, он просто отбрасывается.
        need = {k for k in b0 if k not in ("model", "messages", "provider", "usage", "seed")}
        if (b0.get("response_format") or {}).get("type") == "json_schema":
            # 22.09: DeepInfra заявляет response_format, но НЕ строгую схему — с require_parameters запрос
            # уходил в никуда (404). Просим именно структурированный вывод.
            need.add("structured_outputs")
        cands, info, tps, supported = pick_providers(model, need, cat[model]["out"])
        if ts[0]["job"].get("provider"):                    # провайдер задан в очереди — он и только он
            cands, info = [ts[0]["job"]["provider"]], f"{ts[0]['job']['provider']} (задан в очереди)"
        if not cands:
            log("  подходящих провайдеров нет — выбор оставлен OpenRouter")
        out_est = OUT_EST * (REASONING_OUT_FACTOR if (cat[model].get("reasoning") or {}).get("mandatory") else 1)
        if ts[0]["job"].get("reasoning_effort"):      # включённое размышление: ответ до половины лимита
            out_est = ts[0]["job"].get("max_tokens", MAX_TOKENS_THINK) / 2
        expect = out_est / tps + 60 if tps else 300         # расчётное время ответа: токены ÷ скорость + очередь
        log(f"  кандидаты: {info or 'нет подходящих'} | ждать ответ до {patience([], expect) / 60:.1f} мин")
        tried, times, extra_drop = 0, [], set()            # кандидаты, длительности, параметры-отказники
        bad = []                                          # (разобран ли, обрезан ли) по первым ответам
        log(f"== {model}: {len(ts)} запросов, прогноз ${pr:.2f}, бюджет ${budget}")
        for i, t in enumerate(ts):
            if STOP.exists():
                log("найден logs/cloud.STOP — остановка"); return
            total, by = spent()
            wc = worst_case(cat[model], in_tokens_est(t), job=t["job"])
            ok, why = may_send(total, by.get(model, 0.0), budget, wc)
            if not ok or wc > free - (total - total0):
                log(f"стоп по модели {model}: {why or 'не хватает баланса аккаунта'}"); break
            if first_cost is not None and first_cost * (len(ts) - i) > budget - by.get(model, 0.0):
                log(f"стоп по модели {model}: по факту первого запроса остаток не укладывается в бюджет"); break
            job, rep = t["job"], t["report"]
            tv = job.get("text", "v1")
            report_text = (texts.text_dir(tv) / f"{rep}.txt").read_text()
            if pointless(t, cat):
                continue
            # самолечение: провайдер жалуется на параметр или на схему — убираем ИМЕННО их и шлём тот же
            # запрос снова (до трёх раз), а не пропускаем клетку
            for _heal in range(3):
                ok_params = (supported if supported is not None else cat[model]["params"]) - extra_drop
                body, request_hash = cloud_body(job, report_text, t["run"], cat,
                                                provider or cands[tried:tried + 3], ok_params)
                dropped = body.pop("_dropped")
                started, t0 = now(), time.time()
                r = send(body, patience(times, expect))
                sec = round(time.time() - t0, 1)
                try:
                    resp = r.json()
                except Exception:
                    resp = {"error": {"message": r.text[:300]}}
                if r.status_code != 400:
                    break
                msg = str((resp.get("error") or {}).get("message", "")) + str((resp.get("error") or {}).get("metadata", ""))
                bad_param = next((k for k in ("temperature", "top_p", "seed") if f"`{k}`" in msg), None)
                if "structured_outputs" not in extra_drop and ("grammar" in msg or "schema" in msg.lower()):
                    # Anthropic: «The compiled grammar is too large» — наша схема (24 × 8 полей) ему велика
                    extra_drop.add("structured_outputs")
                    log("  провайдер не принимает нашу схему — повторяю с обычным JSON")
                elif bad_param and bad_param not in extra_drop:
                    # «`temperature` is deprecated for this model» у Opus 5, хотя каталог его показывает
                    extra_drop.add(bad_param)
                    log(f"  провайдер не принимает {bad_param} — убираю и повторяю запрос")
                else:
                    break
            if r.status_code == 402:
                log(f"  402 — на счету или ключе не хватает денег: {str(resp.get('error'))[:200]}. СТОП всего")
                STOP.touch(); return
            if r.status_code != 200 or "choices" not in resp:
                fails += 1
                log(f"  провал {t['name']}: {r.status_code} {str(resp.get('error'))[:200]}")
                if fails >= 2 and tried + 3 < len(cands) + 2 and r.status_code in (0, 429, 502, 503):
                    tried += 1 if provider else 3                  # закреплённый упал — следующий по списку
                    if tried < len(cands):
                        log(f"  смена провайдера: {provider or 'первая тройка'} не отвечает → {cands[tried]}")
                        provider, fails = cands[tried], 0
                        continue
                if fails >= 3:
                    log(f"  три провала подряд — модель {model} снята"); break
                continue
            fails = 0
            times.append(sec)
            usage = resp.get("usage") or {}
            cost = usage.get("cost")
            if cost is None:
                cost = usage.get("prompt_tokens", 0) * cat[model]["in"] + usage.get("completion_tokens", 0) * cat[model]["out"]
            with LEDGER.open("a") as f:
                f.write(json.dumps({"t": now(), "model": model, "name": t["name"], "cost": cost}) + "\n")
            if provider is None:
                provider = resp.get("provider")
                quant = endpoint_quant(model, provider)
                first_cost = cost
                log(f"  провайдер закреплён: {provider}, точность весов: {quant}, первый запрос ${cost:.4f}")
            choice = resp["choices"][0]
            content = choice["message"].get("content") or ""
            meta = {"name": t["name"], "model": model, "cloud": True, "provider": provider, "model_quant": quant,
                    "spec_version": job["spec"], "spec_hash": night.spec_hash(job["spec"]),
                    "request_hash": request_hash, "report": rep, "text_version": tv,
                    "text_hash": night.text_hashes(tv)[rep]["sha256"], "run": t["run"],
                    "variant": night.variant(job), "temperature": body.get("temperature"), "top_p": body.get("top_p"),
                    "seed": body.get("seed"), "min_p_dropped": "min_p" in job, "schema": "response_format" in body,
                    "dropped_params": dropped, "max_tokens": body["max_tokens"], "reasoning_mode": reasoning_setup(cat[model], job)[2],
                    "or_cache": r.headers.get("X-OpenRouter-Cache-Status"), "ids_order": job.get("ids_order", "spec"), "mode": job.get("mode", "score"),
                    "started": started, "finished": now(), "sec": sec, "cost_usd": cost,
                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    "finish_reason": choice.get("finish_reason"),
                    "truncated": choice.get("finish_reason") in ("length", "error"),
                    "parse_ok": night.quick_parse(content), "generation_id": resp.get("id"),
                    "cache_contaminated": False}
            tmp = RAW / f"{t['name']}.json.tmp"
            tmp.write_text(json.dumps({"meta": meta, "content": content, "response": resp}, ensure_ascii=False, indent=1))
            tmp.rename(RAW / f"{t['name']}.json")
            requests_done += 1
            bad.append(not meta["parse_ok"] or meta["truncated"] or filled(content) < 0.3)
            if len(bad) == 3 and sum(bad) == 3:
                log(f"  стоп по модели {model}: первые три ответа пустые или не разбираются — деньги не тратим")
                break
            if len(bad) == 5 and sum(bad) >= 2:
                log(f"  стоп по модели {model}: из первых 5 ответов плохих {sum(bad)} (не разобран или обрезан) — деньги не тратим")
                break
            total, by = spent()
            log(f"  ✓ {t['name'][-48:]} {sec}с ${cost:.4f} | модель ${by[model]:.2f}/{budget} | всего ${total:.2f}/{GLOBAL_CAP}"
                f" | json={meta['parse_ok']}")
            if requests_done % 10 == 0:                      # сверка с OpenRouter: их счётчик главнее нашего
                try:
                    usage_key = get("/key")["data"].get("usage", 0)
                except Exception:
                    usage_key = 0
                    log("  сверка с OpenRouter не удалась (сеть) — продолжаю по своему журналу")
                if usage_key >= GLOBAL_CAP:
                    log(f"OpenRouter показывает ${usage_key:.2f} — общий стоп"); return
    total, _ = spent()
    log(f"готово: потрачено этим раннером ${total:.2f}")


# ---------- цепочка: модели строго по одной, после каждой — разбор ----------

GO, STATUS = LOGS / "cloud.GO", LOGS / "cloud_status.json"
AUTO_CONTINUE_MIN = 40      # если разбора нет столько минут, а автопроверка чистая — идём дальше сами


def model_gate(model):
    """Автопроверка готовой модели по её ответам: доля разобранных ≥ 80%, обрезанных ≤ 10%."""
    metas = [json.loads(f.read_text())["meta"] for f in RAW.glob(f"{night.safe(model)}__*.json")]
    if not metas:
        return False, "ответов нет"
    ok = sum(bool(m.get("parse_ok")) for m in metas) / len(metas)
    tr = sum(bool(m.get("truncated")) for m in metas) / len(metas)
    return ok >= 0.8 and tr <= 0.1, f"ответов {len(metas)}, разобрано {ok:.0%}, обрезано {tr:.0%}"


def status(**kw):
    """Пульс цепочки. pid — чтобы сторож проверял именно этот процесс, а не совпадение по тексту команды."""
    STATUS.write_text(json.dumps({"t": now(), "pid": os.getpid(), **kw}, ensure_ascii=False))


def chain(queue):
    GO.unlink(missing_ok=True)
    done = []
    while True:
        cat = catalog()      # модели, у которых остались только бессмысленные серии (T>0 без температуры), — мимо
        todo = [m for m, ts in tasks_by_model(queue).items()
                if m not in done and m in cat and any(not pointless(t, cat) for t in ts)]
        if not todo or STOP.exists():
            break
        model = todo[0]
        status(state="running", model=model)
        log(f"=== цепочка: {model}")
        run(queue, only=model)
        done.append(model)
        if STOP.exists():
            break
        passed, why = model_gate(model)
        total, _ = spent()
        status(state="review", model=model, gate=passed, gate_info=why, spent=round(total, 3))
        log(f"=== {model} готова: автопроверка {'чистая' if passed else 'НЕ ПРОЙДЕНА'} ({why}); жду разбора")
        t0 = time.time()
        while not GO.exists() and not STOP.exists():
            if passed and time.time() - t0 > AUTO_CONTINUE_MIN * 60:
                log(f"=== разбора нет {AUTO_CONTINUE_MIN} мин, автопроверка чистая — продолжаю сам"); break
            time.sleep(20)
        GO.unlink(missing_ok=True)
    total, _ = spent()
    status(state="stopped" if STOP.exists() else "finished", spent=round(total, 3))
    log(f"=== цепочка завершена: потрачено ${total:.2f}")


def selftest():
    prices = {"in": 10e-6, "out": 50e-6}
    wc = worst_case(prices, 16000)
    assert abs(wc - (0.16 + 0.45)) < 1e-9, wc
    assert may_send(17.5, 0, 5, wc)[0] is False                       # общий стоп
    assert may_send(1.0, 4.6, 5, wc)[0] is False                      # бюджет модели
    assert may_send(1.0, 1.0, 5, wc)[0] is True
    assert NetFail().status_code == 0 and "error" in NetFail().json()
    base = {"in": 1e-6, "out": 1e-6}
    assert filled('[{"id": "a", "q_value": "1"}, {"id": "b"}]') == 0.5
    assert filled('{"a": {"q_value": "1"}, "b": {}}') == 0.5 and filled("не json") == 0.0
    assert patience([], 30) == 180 and patience([], 400) == 900 and patience([], 100) == 300
    assert patience([40, 45, 50]) == 180 and patience([200, 300, 400]) == 900 and patience([60, 70, 80]) == 280
    assert reasoning_setup({**base, "params": set()})[0] is None
    assert reasoning_setup(base, {"reasoning_effort": "high"})[1] == MAX_TOKENS_THINK
    assert reasoning_setup({**base, "params": {"reasoning"}, "reasoning": {"mandatory": False}})[0] == {"enabled": False}
    r, mt, _ = reasoning_setup({**base, "params": {"reasoning"},
                                "reasoning": {"mandatory": True, "supported_efforts": ["high", "medium", "low"]}})
    assert r == {"effort": "low", "exclude": True} and mt == MAX_TOKENS_REASONING
    print("cloud: самопроверка пройдена")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    {"plan": lambda: plan(sys.argv[2]), "run": lambda: run(sys.argv[2], only), "chain": lambda: chain(sys.argv[2]),
     "selftest": selftest}[cmd]()
