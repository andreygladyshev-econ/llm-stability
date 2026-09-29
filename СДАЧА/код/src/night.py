"""Ночной раннер: очередь заданий → raw/*.json.

Команды:
  night.py check  queue.toml   предпроверка, ничего не запускает
  night.py start  queue.toml   предпроверка → фоновый запуск под caffeinate, можно закрыть терминал
  night.py run    queue.toml   то же, но в этом окне (для отладки)
  night.py status              что происходит сейчас
  night.py pause / resume      пауза после текущего прогона (модель выгружается, память свободна) / продолжить
  night.py until 09:50         сменить время остановки прямо ночью
  night.py stop                мягкая остановка: доделать текущий прогон, собрать отчёт

Очередь перечитывается перед каждым прогоном: в неё можно дописывать задания посреди ночи.

Порядок: задание (как в файле) → номер прогона → отчёт. Каждый ответ пишется на диск сразу;
готовые файлы пропускаются, поэтому повторный запуск продолжает с места остановки.

Барьеры против «чужой памяти» (почему прогоны независимы):
  1. Перед КАЖДЫМ прогоном модель выгружается и загружается заново. Кэш промпта в LM Studio
     живёт до выгрузки модели; 2026-09-14 кэш перевернул знак оценки при temperature=0.
  2. После прогона лог сервера проверяется: cached_tokens должен быть 0, иначе прогон помечен
     cache_contaminated и в отчёте считается отдельно.
  3. Перед запросом проверяется, что загружена ровно одна нужная модель (JIT-загрузка не подменит настройки).
  4. Запрос без истории, без инструментов; спекулятивное декодирование выключено; один запрос за раз.
  5. В каждый результат пишется отпечаток окружения: версии движков, macOS, код, спека, текст, модель.
"""
import datetime as dt
import hashlib
import json
import os
import platform
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from indicators import IDS  # noqa: E402
import texts  # noqa: E402
import score_rules  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANDREY = ROOT.parent / "_АНДРЕЙ"  # папка Андрея лежит рядом с проектом, не внутри
RAW, TEXT, SPECS, LOGS = ROOT / "raw", ROOT / "text", ROOT / "specs", ROOT / "logs"
FAILED = RAW / "_failed"
STATUS, LOCK = LOGS / "status.json", LOGS / "night.lock"
PAUSE, STOP, UNTIL_FILE = LOGS / "PAUSE", LOGS / "STOP", LOGS / "until.txt"
LMS = str(Path.home() / ".lmstudio/bin/lms")
SERVER_LOGS = Path.home() / ".lmstudio/server-logs"
API = "http://localhost:50255/v1/chat/completions"
# 18.09 LM Studio сгруппировал варианты Qwen 27B под ключом qwen/qwen3.8-27b, старый ключ пропал.
# Имя в очереди и в файлах прежнее (преемственность с ночами 1–4); грузим по новому ключу, API-имя задаём
# через --identifier и после загрузки сверяем вариант: переключат в LM Studio на 5bit — прогон упадёт, а не подменится.
# имя в задании → (ключ модели в LM Studio, ожидаемый вариант). Разные сжатия и движки одной модели —
# H11 (движок) и H6b (глубина сжатия): всё это один и тот же Qwen 27B, отличается только упаковка весов.
MODEL_LOAD = {"qwen3.8-27b-mlx@4bit": ("qwen/qwen3.8-27b", "qwen/qwen3.8-27b@4bit"),
              # 21.09: вариант карточки нельзя выбрать из командной строки, но его можно переключить в окне
              # LM Studio («Switch model source»). Тогда грузим обычным способом, а проверка варианта
              # не даст перепутать: если Андрей не переключил источник, прогон остановится с ошибкой.
              "qwen3.8-27b-mlx@5bit": ("qwen/qwen3.8-27b", "qwen/qwen3.8-27b@5bit"),
              "qwen3.8-27b-gguf@q4km": ("qwen/qwen3.8-27b", "qwen/qwen3.8-27b@q4_k_m"),
              "qwen3.8-27b-gguf@q6k": ("qwen/qwen3.8-27b", "qwen/qwen3.8-27b@q6_k")}
# 21.09: `lms load` умеет грузить только вариант карточки по умолчанию («Model not found» на остальных),
# а сервер по API поднимает любой вариант сам. Для таких моделей грузим запросом-разогревом.

MEM_LIMIT_GIB = 24          # CLAUDE.md: выше — риск зависания
MIN_FREE_PCT = 8            # ниже — Mac уже упирается в память
SWAP_GROWTH_MB = 1500       # рост свопа за один прогон — модель не влезла
LOW_BATTERY_PCT = 25
ATTEMPTS = 3
FAILS_TO_SKIP_MODEL = 3     # столько провалов подряд — модель пропускаем до конца ночи
NO_THINK = "<think>\n\n</think>\n\n"  # пустой черновик размышлений: у MLX-Qwen иначе не выключить
# L1 «два хода» (литература: Format Tax, Tam 2024): сначала свободный текст, затем перенос в JSON по схеме
TURN1_NOTE = ("\n\n=== ФОРМАТ ЭТОГО ШАГА ===\nСейчас JSON не нужен. Для каждого из 24 показателей по порядку напиши "
              "обычным текстом одну короткую запись: id — какие цифры и за какой период сравнил — какое правило "
              "применил — дословная цитата — балл от -2 до 2.")
TURN2_ASK = "Перенеси свои оценки в JSON строго по схеме. Баллы, цитаты и обоснования не меняй."


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def log(*a):
    print(now(), *a, flush=True)


def sha(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def sh(*cmd, timeout=600):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).replace("\r", "\n")


# ---------- очередь ----------

def variant(job):
    v = f"t{job.get('temperature', 0)}"
    if job.get("think"):
        v += "-think"
    if job.get("repeat_penalty", 1.0) != 1.0:
        v += f"-rp{job['repeat_penalty']}"
    if job.get("ids_order") == "shuffled":
        v += "-shuf"
    if job.get("prefill", "auto") is False:
        v += "-nopf"
    if job.get("schema", True) is False:
        v += "-nojs"
    if not job.get("fresh_load", True):
        v += "-warm"
    if job.get("order") == "report_first":
        v += "-rf"
    if job.get("fields") == "score_first":
        v += "-sf"
    if job.get("two_turn"):
        v += "-2t"
    if job.get("mode") == "extract":
        v += "-ex"
    if job.get("mode") == "extract2":
        v += "-ex2"
    if "min_p" in job:
        v += f"-mp{job['min_p']}"
    if job.get("tag"):                  # явная метка серии, когда отличие не видно по остальным полям (напр. контекст)
        v += f"-{job['tag']}"
    if job.get("text", "v1") != "v1":
        v += f"-tx{job['text'][1:]}"  # другая версия текста отчёта (src/texts.py)
    return v


def job_key(t):
    return f"{t['job']['model']} {t['job']['spec']} {variant(t['job'])}"


def safe(s):
    return re.sub(r"[^\w.-]+", "-", s)


def load_queue(path):
    q = tomllib.loads(Path(path).read_text())
    tasks = []
    for prio, job in enumerate(q["job"]):
        # сначала прогон 1 по всем отчётам, потом прогон 2… — при остановке в любой момент
        # число прогонов по отчётам различается максимум на один (иначе метрики несравнимы, см. selfaudit)
        for n in range(1, job["runs"] + 1):
            for rep in job["reports"]:
                name = f"{safe(job['model'])}__{job['spec']}__{variant(job)}__{rep}__run{n}"
                tasks.append(dict(job=job, prio=prio, report=rep, run=n, name=name))
    return q, tasks


def model_order(tasks):
    """Внешний цикл — модель. Модели идут в порядке первого появления в очереди."""
    seen = []
    for t in tasks:
        if t["job"]["model"] not in seen:
            seen.append(t["job"]["model"])
    return seen


def pending(tasks):
    return [t for t in tasks if not (RAW / f"{t['name']}.json").exists()]


# ---------- спецификация, текст, окружение: хеши ----------

def spec_bundle(version):
    return (SPECS / f"{version}.md").read_text(), (SPECS / f"{version}.prompt.md").read_text()


def spec_hash(version):
    spec, prompt = spec_bundle(version)
    return sha(spec + "\0" + prompt)


def text_hashes(version="v1"):
    return texts.hashes(version)


def check_spec_consistency(versions):
    """Спеку отредактировали, а версию не подняли → стоп."""
    problems = []
    for v in versions:
        h = spec_hash(v)
        for f in RAW.glob(f"*__{v}__*.json"):
            if json.loads(f.read_text())["meta"]["spec_hash"] != h:
                problems.append(f"{v}: файл спеки изменился после прогонов ({f.name}). Поднимите номер версии.")
                break
    return problems


def environment(queue_path):
    """Отпечаток всего, что может незаметно изменить ответ модели."""
    backends = {}
    p = Path.home() / ".lmstudio/.internal/backend-preferences-v1.json"
    if p.exists():
        backends = {b["model_format"]: f"{b['name']} {b['version']}" for b in json.loads(p.read_text())}
    _, lms_ver = sh(LMS, "version", timeout=30)
    code = "".join((ROOT / "src" / f).read_text() for f in ("night.py", "indicators.py"))
    _, commit = sh("git", "-C", str(ROOT), "rev-parse", "--short", "HEAD", timeout=30)
    lms_commit = re.search(r"CLI commit:\s*(\w+)", re.sub(r"\x1b\[[0-9;]*m", "", lms_ver))
    return {"backends": backends, "lms_cli_commit": lms_commit.group(1) if lms_commit else None,
            "macos": platform.mac_ver()[0], "python": platform.python_version(),
            "code_sha": sha(code)[:16], "git_commit": commit.strip() if len(commit.strip()) < 20 else None,
            "queue_sha": sha(Path(queue_path).read_text())[:16]}


def env_drift(env, tasks):
    """Движок обновился между ночами → результаты одной конфигурации несопоставимы."""
    drift = []
    for m in model_order(tasks):
        for f in RAW.glob(f"{safe(m)}__*.json"):
            old = json.loads(f.read_text())["meta"].get("env", {}).get("backends")
            if old and old != env["backends"]:
                drift.append(f"{m}: версии движков изменились с прошлых прогонов ({old} → {env['backends']})")
            break
    return drift


# ---------- система ----------

def mem_free_pct():
    _, out = sh("memory_pressure", timeout=30)
    m = re.search(r"free percentage:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def swap_used_mb():
    _, out = sh("sysctl", "vm.swapusage")
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else None


def on_ac():
    _, out = sh("pmset", "-g", "ps")
    return "AC Power" in out


def power():
    """Батарея, мощность зарядки и потребление. Слабая зарядка под нагрузкой = батарея тает всю ночь."""
    _, out = sh("ioreg", "-rn", "AppleSmartBattery")
    _, therm = sh("pmset", "-g", "therm")

    def num(key):
        m = re.search(rf'"{key}"\s*=\s*(\d+)', out)
        v = int(m.group(1)) if m else None
        return None if v is not None and v >= 2**63 else v  # на батарее ioreg отдаёт отрицательное как 2^64−x

    # сколько записано на SSD с загрузки системы: MLX пишет кэш промпта на диск, меряем износ
    _, disk = sh("ioreg", "-c", "IOBlockStorageDriver", "-r", "-k", "Statistics")
    written = [int(x) for x in re.findall(r'"Bytes \(Write\)"=(\d+)', disk)]
    return {"ssd_written_gb": round(max(written) / 1e9, 1) if written else None,
            "battery_pct": num("CurrentCapacity"), "adapter_w": num("Watts"),
            "power_in_w": round((num("SystemPowerIn") or 0) / 1000), "load_w": round((num("SystemLoad") or 0) / 1000),
            "thermal_warning": "No thermal warning" not in therm}


def server_up():
    try:
        httpx.get(API.replace("chat/completions", "models"), timeout=5)
        return True
    except httpx.HTTPError:
        return False


def ensure_server():
    if server_up():
        return True
    log("сервер LM Studio не отвечает — запускаю")
    sh(LMS, "server", "start", "--port", "50255", timeout=120)
    for _ in range(12):
        time.sleep(5)
        if server_up():
            return True
    return False


def downloaded_models():
    _, out = sh(LMS, "ls", "--json")
    d = {m["modelKey"]: m for m in json.loads(out[out.index("["):])}
    for label, (key, _) in MODEL_LOAD.items():
        base = key.split("@")[0]        # варианты сжатия перечислены внутри одной карточки модели
        if key in d or base in d:
            d[label] = d.get(key, d.get(base))
    return d


def estimate_gib(model, ctx):
    _, out = sh(LMS, "load", MODEL_LOAD.get(model, (model,))[0], "-c", str(ctx), "--estimate-only", "-y", timeout=120)
    m = re.search(r"Estimated Total Memory:\s*([\d.]+)\s*GiB", out)
    return float(m.group(1)) if m else None


def loaded_models():
    _, out = sh(LMS, "ps", "--json")
    try:
        return {m.get("identifier") or m.get("modelKey"): m for m in json.loads(out[out.index("["):])}
    except (ValueError, json.JSONDecodeError):
        return {}


def jit_load(model, ctx):
    """Загрузка вариантом, который не умеет грузить CLI: короткий запрос — сервер поднимает модель сам."""
    httpx.post(API, json={"model": model, "messages": [{"role": "user", "content": "ок"}],
                          "max_tokens": 1, "temperature": 0}, timeout=1800)
    loaded = loaded_models()
    if list(loaded) != [model]:
        raise RuntimeError(f"после разогрева в памяти {list(loaded)}, а нужен только {model}")
    return loaded[model]


def fresh_load(model, ctx):
    """Барьер 1: выгрузить всё и загрузить модель заново — кэш промпта пуст."""
    sh(LMS, "unload", "--all", timeout=120)
    key, variant_ = MODEL_LOAD.get(model, (model, None))
    if "@" in key and key == model:     # вариант карточки: CLI его не находит, поднимаем через сервер
        return jit_load(model, ctx)
    for flags in (["--no-speculative-draft-mtp"], []):  # флаг есть не у всех форматов
        code, out = sh(LMS, "load", key, "-c", str(ctx), "--parallel", "1", "--identifier", model, *flags, "-y",
                       timeout=900)
        if code == 0:
            loaded = loaded_models()
            if list(loaded) == [model]:
                got = loaded[model].get("selectedVariant")
                if variant_ and got != variant_:
                    raise RuntimeError(f"загружен вариант {got}, нужен {variant_} — проверьте выбор варианта в LM Studio")
                return loaded[model]
            out = f"после загрузки в памяти не только {model}: {list(loaded)}"
    raise RuntimeError(f"модель не загрузилась: {out[-300:]}")


def server_log_facts(since, until):
    """Барьер 2: что сервер сам пишет о кэше и контексте в окне [since, until] одного прогона."""
    files = sorted(SERVER_LOGS.glob("*/*.log"), key=lambda p: p.stat().st_mtime)
    if not files:
        return {"cache_cached_tokens": None, "cache_uncached_tokens": None, "ctx_effective": None}
    # два последних файла: в полночь сервер начинает новый лог, прогон может прийтись на стык
    lines = [ln for f in files[-2:] for ln in f.read_text(errors="replace").splitlines()[-3000:]]
    lo, hi = since.replace("T", " "), until.replace("T", " ")
    cached = uncached = ctx = None
    for line in lines:
        ts = line[1:20] if line.startswith("[") else None
        if m := re.search(r"context target: configured=([\d,]+) fitted=([\d,]+) effective=([\d,]+)", line):
            ctx = int(m.group(3).replace(",", ""))
        if ts and lo <= ts <= hi and (m := re.search(r"Prompt cache restore: cached_tokens=(\d+) uncached_tokens=(\d+)", line)):
            cached = (cached or 0) + int(m.group(1))
            uncached = (uncached or 0) + int(m.group(2))
    return {"cache_cached_tokens": cached, "cache_uncached_tokens": uncached, "ctx_effective": ctx}


# ---------- один прогон ----------

def response_schema(score_first=False, ids=IDS):
    # порядок полей = порядок, в котором модель их пишет при строгом JSON
    order = ["score", "quote", "reasoning"] if score_first else ["quote", "reasoning", "score"]
    fields = {"quote": {"type": "string"}, "reasoning": {"type": "string"},
              "score": {"type": "integer", "enum": [-2, -1, 0, 1, 2]}}
    item = {"type": "object", "additionalProperties": False, "required": order,
            "properties": {k: fields[k] for k in order}}
    return {"type": "object", "additionalProperties": False, "required": list(ids),
            "properties": {i: item for i in ids}}


def job_ids(job, run):
    # порядок показателей: как в спеке или перемешан (одна перестановка на номер прогона, одинаковая для всех отчётов)
    ids = list(IDS)
    if job.get("ids_order") == "shuffled":
        # order_seed фиксирует перестановку независимо от номера прогона: нужно для разложения разброса
        # «порядок × повтор» (2608.16253) — несколько повторов при одном и том же порядке
        random.Random(f"ids-{job.get('order_seed', run)}").shuffle(ids)
    return ids


def schema_format(job, ids):
    return {"type": "json_schema", "json_schema": {"name": "scores", "strict": True,
                                                   "schema": response_schema(job.get("fields") == "score_first", ids)}}


def extract_schema(ids):
    """H3: модель выписывает числа и цитаты, баллы числовых показателей считает Python (src/score_rules.py)."""
    order = ["quote", "period", "value", "base", "change", "score"]
    fields = {k: {"type": "string"} for k in order[:-1]}
    fields["score"] = {"type": "integer", "enum": [-2, -1, 0, 1, 2]}
    item = {"type": "object", "additionalProperties": False, "required": order, "properties": fields}
    return {"type": "object", "additionalProperties": False, "required": list(ids),
            "properties": {i: item for i in ids}}


def extract2_schema(ids):
    """v7: у числовых показателей две ячейки периода (q — квартал, y — с начала года), выбирает код."""
    num = [f"{p}_{k}" for p in ("q", "y") for k in ("quote", "value", "base", "change")] + ["score"]
    judg = ["quote", "score"]
    sc = {"type": "integer", "enum": [-2, -1, 0, 1, 2]}
    item = lambda order: {"type": "object", "additionalProperties": False, "required": order,  # noqa: E731
                          "properties": {k: sc if k == "score" else {"type": "string"} for k in order}}
    return {"type": "object", "additionalProperties": False, "required": list(ids),
            "properties": {i: item(num if i in score_rules.NUMERIC else judg) for i in ids}}


def build_request(job, report_text, run=1):
    ids = job_ids(job, run)
    spec, prompt = spec_bundle(job["spec"])
    report_block = "=== ТЕКСТ ОТЧЁТА ===\n" + report_text
    if job.get("order") == "report_first":
        # отчёт в начале, правила — сразу перед ответом
        system = prompt.replace("{ids}", ", ".join(ids)).replace("{spec}", "(спецификация — после текста отчёта)")
        user = report_block + "\n\n=== СПЕЦИФИКАЦИЯ ===\n" + spec
    else:
        system = prompt.replace("{ids}", ", ".join(ids)).replace("{spec}", spec)
        user = report_block
    if job.get("two_turn"):
        user += TURN1_NOTE
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    t = job.get("temperature", 0)
    body = {"model": job["model"], "messages": messages, "temperature": t,
            "top_p": 1, "top_k": 1 if t == 0 else 40,
            # штраф за повтор слов мешает дословно цитировать отчёт → по умолчанию выключен
            "repeat_penalty": job.get("repeat_penalty", 1.0),
            "max_tokens": job.get("max_tokens", 24000 if job.get("think") else 8000), "stream": False}
    # 19.09: с новой карточкой модели LM Studio подставляет свои умолчания выборки (min_p выключен).
    # Не заданный в запросе параметр — скрытая переменная; задаём явно, когда он указан в задании.
    if "min_p" in job:
        body["min_p"] = job["min_p"]
    if not job.get("think"):
        # строгий JSON совместим только с выключенным размышлением (проверено 2026-09-14).
        # Пустой <think> понимают только модели семейства Qwen; остальным он сломал бы ответ.
        prefill = job.get("prefill", "auto")  # auto / true / false — у T-pro шаблон сам вставляет <think>
        # только MLX: в llama.cpp (GGUF) подстановка + JSON-схема = 400 «Failed to initialize samplers» (T-pro, Qwen 4B, 16.09)
        if prefill is True or (prefill == "auto" and any(k in job["model"].lower() for k in ("mlx", "bonsai"))):
            messages.append({"role": "assistant", "content": NO_THINK})
    if not job.get("think") and job.get("schema", True) and not job.get("two_turn"):
        body["response_format"] = ({"type": "json_schema",
                                    "json_schema": {"name": "extract", "strict": True, "schema": extract_schema(ids)}}
                                   if job.get("mode") == "extract" else
                                   {"type": "json_schema",
                                    "json_schema": {"name": "extract2", "strict": True, "schema": extract2_schema(ids)}}
                                   if job.get("mode") == "extract2" else schema_format(job, ids))
    extra = TURN2_ASK if job.get("two_turn") else ""
    return body, sha(json.dumps(body, ensure_ascii=False, sort_keys=True) + extra)


def followup_request(job, body, draft, run):
    """Второй ход: тот же диалог + черновик модели + просьба перенести в JSON по схеме."""
    msgs = [m for m in body["messages"] if not (m["role"] == "assistant" and m["content"] == NO_THINK)]
    msgs += [{"role": "assistant", "content": draft}, {"role": "user", "content": TURN2_ASK}]
    if body["messages"][-1]["content"] == NO_THINK:
        msgs.append({"role": "assistant", "content": NO_THINK})
    return {**body, "messages": msgs, "response_format": schema_format(job, job_ids(job, run))}


def quick_parse(content):
    """Только для счётчиков в статусе. Полный разбор — в report.py.

    23.09: Opus 5 отвечает списком объектов с «id» в обёртке ```json — раньше считалось «не разобрано»
    и облачная защита снимала модель. Понимаем оба вида, как и report.parse."""
    i, j = content.find("["), content.find("{")
    try:
        if 0 <= i < (j if j >= 0 else len(content)):
            d = {x.get("id"): x for x in json.loads(content[i:content.rindex("]") + 1]) if isinstance(x, dict)}
        else:
            d = json.loads(content[j:content.rindex("}") + 1])
    except (ValueError, TypeError, AttributeError):
        return False
    # обёртка бывает любой: «items», «indicators», просто список (Opus 5 меняет форму от запроса к запросу)
    if isinstance(d, dict) and not any(i in d for i in IDS):
        lst = next((v for v in d.values() if isinstance(v, list) and any(isinstance(x, dict) for x in v)), None)
        if lst is not None:
            d = {x.get("id"): x for x in lst if isinstance(x, dict)}
    return isinstance(d, dict) and all(i in d for i in IDS)


def run_task(t, model_info, env):
    job, rep = t["job"], t["report"]
    ctx = job.get("ctx", 32768)
    load_t0 = time.time()
    if job.get("fresh_load", True) or list(loaded_models()) != [job["model"]]:
        loaded_info = fresh_load(job["model"], ctx)
    else:
        # контрольное условие для гипотезы H1: модель НЕ перезагружается, кэш промпта разрешён
        loaded_info = loaded_models()[job["model"]]
    load_sec = round(time.time() - load_t0, 1)

    tv = job.get("text", "v1")
    text_file = texts.text_dir(tv) / f"{rep}.txt"
    report_text = text_file.read_text()
    if sha(report_text.encode()) != text_hashes(tv)[rep]["sha256"]:
        raise RuntimeError(f"текст {rep} изменился после заморозки")
    body, request_hash = build_request(job, report_text, t["run"])
    if list(loaded_models()) != [job["model"]]:  # барьер 3
        raise RuntimeError("перед запросом загружена не та модель")

    timeout = job.get("timeout_min", 60 if job.get("think") else 30) * 60
    def ask(b):
        r = httpx.post(API, json=b, timeout=httpx.Timeout(timeout, connect=10))
        r.raise_for_status()
        return r.json()

    swap0, pw0, started, t0 = swap_used_mb(), power(), now(), time.time()
    resp, first = ask(body), None
    if job.get("two_turn"):
        first, t1_end = resp, now()
        time.sleep(2)  # строки лога второго хода не попадут в окно первого
        facts = server_log_facts(started, t1_end)
        t2_start = now()
        resp = ask(followup_request(job, body, first["choices"][0]["message"].get("content") or "", t["run"]))
    sec = round(time.time() - t0, 1)
    choice = resp["choices"][0]
    msg = choice["message"]
    usage = dict(resp.get("usage", {}))
    time.sleep(2)  # сервер дописывает строки лога после ответа
    if first:
        # второй ход законно берёт из кэша свой же первый ход; чистоту проверяем по первому
        facts["turn2_cached_tokens"] = server_log_facts(t2_start, now())["cache_cached_tokens"]
        u1 = first.get("usage", {})
        for k in ("prompt_tokens", "completion_tokens"):
            usage[k] = (usage.get(k) or 0) + (u1.get(k) or 0)
    else:
        facts = server_log_facts(started, now())
    meta = {
        "name": t["name"], "model": job["model"], "model_format": model_info.get("format"),
        "model_arch": model_info.get("architecture"), "model_path": model_info.get("path"),
        "model_size_bytes": model_info.get("sizeBytes"), "model_quant": (model_info.get("quantization") or {}).get("name"),
        "ctx_requested": ctx, "ctx_loaded": loaded_info.get("contextLength"), **facts,
        "cache_contaminated": bool(facts["cache_cached_tokens"]),
        "spec_version": job["spec"], "spec_hash": spec_hash(job["spec"]), "request_hash": request_hash,
        "report": rep, "text_version": tv, "text_hash": text_hashes(tv)[rep]["sha256"], "run": t["run"],
        "variant": variant(job), "temperature": body["temperature"], "top_k": body["top_k"], "top_p": body["top_p"],
        "repeat_penalty": body["repeat_penalty"], "think": bool(job.get("think")),
        "schema": "response_format" in body or bool(first), "max_tokens": body["max_tokens"],
        "prefill": body["messages"][-1]["role"] == "assistant", "ids_order": job.get("ids_order", "spec"),
        "mode": job.get("mode", "score"),
        "started": started, "finished": now(), "sec": sec, "load_sec": load_sec,
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "tok_per_sec": round(usage.get("completion_tokens", 0) / sec, 1) if sec else None,
        "finish_reason": choice.get("finish_reason"),
        "truncated": choice.get("finish_reason") == "length"
                     or bool(first and first["choices"][0].get("finish_reason") == "length"),
        "two_turn": bool(first), "draft_tokens": (first or {}).get("usage", {}).get("completion_tokens"),
        "swap_before_mb": swap0, "swap_after_mb": swap_used_mb(), "mem_free_pct_after": mem_free_pct(),
        "battery_pct_before": pw0["battery_pct"], **power(),
        "parse_ok": quick_parse(msg.get("content") or ""), "env": env,
    }
    record = {"meta": meta, "content": msg.get("content") or "",
              "reasoning": msg.get("reasoning_content") or "", "response": resp,
              "draft": (first["choices"][0]["message"].get("content") or "") if first else None,
              "draft_response": first}
    tmp = RAW / f"{t['name']}.json.tmp"
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    tmp.rename(RAW / f"{t['name']}.json")  # атомарно: оборванный файл не выдаст себя за готовый
    (FAILED / f"{t['name']}.json").unlink(missing_ok=True)
    return meta


# ---------- статус ----------

class Status:
    def __init__(self, queue_path, tasks, env):
        self.s = {"queue": str(queue_path), "pid": os.getpid(), "started": now(), "env": env,
                  "total": len(tasks), "already_done": len(tasks) - len(pending(tasks)), "done": 0, "failed": 0,
                  "truncated": 0, "parse_failed": 0, "cache_contaminated": 0, "current": None,
                  "last_error": None, "skipped_models": {}, "sec_by_model": {}, "sec_by_job": {},
                  "paused": False, "events": []}
        self.save()

    def event(self, text):
        log(text)
        self.s["events"].append(f"{now()} {text}")
        self.save()

    def save(self):
        self.s["updated"] = now()
        LOGS.mkdir(exist_ok=True)
        tmp = STATUS.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.s, ensure_ascii=False, indent=1))
        tmp.rename(STATUS)
        self.save_human()

    def save_human(self):
        """То же простым языком → _АНДРЕЙ/СЕЙЧАС.md (Андрей смотрит этот файл)."""
        s = self.s
        left = s.get("left", s["total"] - s["already_done"] - s["done"] - s["failed"])
        state = ("завершена " + s["finished"] if s.get("finished")
                 else "ПАУЗА (продолжить: night.py resume)" if s.get("paused") else "идёт")
        lines = [f"# Ночь сейчас: {state}", "",
                 f"- очередь: `{s['queue']}`, старт {s['started']}, работать до {s.get('until') or 'конца очереди'}",
                 f"- обновлено: {s['updated']}",
                 f"- сделано {s['done']}, осталось {left}, провалов {s['failed']}"
                 f" (было готово заранее: {s['already_done']})",
                 f"- сейчас: {s['current'] or '—'}",
                 f"- обрезанных ответов {s['truncated']}, неразобранных {s['parse_failed']}, "
                 f"с чужим кэшем {s['cache_contaminated']} (должно быть 0)",
                 f"- последняя ошибка: {s['last_error'] or 'нет'}",
                 f"- пропущенные модели: {s['skipped_models'] or 'нет'}", "",
                 "## Последние события", "", *[f"- {e}" for e in s["events"][-15:]], ""]
        out = ANDREY / "СЕЙЧАС.md"
        tmp = out.with_suffix(".tmp")
        tmp.write_text("\n".join(lines))
        tmp.rename(out)


def median_sec(model, think):
    secs = []
    for f in RAW.glob(f"{safe(model)}__*.json"):
        m = json.loads(f.read_text())["meta"]
        if m["think"] == bool(think):
            secs.append(m["sec"] + m.get("load_sec", 20))
    return statistics.median(secs) if secs else None


# ---------- предпроверка ----------

def check(queue_path):
    q, tasks = load_queue(queue_path)
    todo = pending(tasks)
    ok, warn, fail = [], [], []
    if not ensure_server():
        fail.append("сервер LM Studio не запускается")
    if not ANDREY.is_dir():
        fail.append(f"нет папки Андрея {ANDREY} — туда пишутся СЕЙЧАС.md и отчёт; если её перенесли, поправить ANDREY")
    if not on_ac():
        warn.append("Mac не на зарядке")
    pw = power()
    if (pw["adapter_w"] or 0) < 90:
        warn.append(f"зарядка {pw['adapter_w']} Вт: под нагрузкой Mac берёт ~90 Вт, батарея будет разряжаться — "
                    "подключите штатный MagSafe")
    free = mem_free_pct()
    if free is not None and free < 25:
        warn.append(f"свободно памяти {free}% — закройте браузер и лишние программы")
    loaded = loaded_models()
    if loaded:
        warn.append(f"сейчас загружены модели {list(loaded)} — перед стартом будут выгружены")
    env = environment(queue_path)
    warn += env_drift(env, tasks)
    models = downloaded_models()
    for rep, tv in sorted({(t["report"], t["job"].get("text", "v1")) for t in tasks}):
        p, hashes = texts.text_dir(tv) / f"{rep}.txt", text_hashes(tv)
        if not p.exists() or rep not in hashes:
            fail.append(f"нет замороженного текста {rep} (версия {tv})")
        elif sha(p.read_bytes()) != hashes[rep]["sha256"]:
            fail.append(f"текст {rep} (версия {tv}) изменился после заморозки — прогоны на нём несопоставимы")
    for v in sorted({t["job"]["spec"] for t in tasks}):
        if not (SPECS / f"{v}.md").exists() or not (SPECS / f"{v}.prompt.md").exists():
            fail.append(f"нет файлов спеки {v}.md / {v}.prompt.md")
    if not fail:
        fail += check_spec_consistency({t["job"]["spec"] for t in tasks})
    total_h = 0
    for m in model_order(todo):
        jobs = [t["job"] for t in todo if t["job"]["model"] == m]
        if m not in models:
            fail.append(f"модель {m} не скачана")
            continue
        ctx = max(j.get("ctx", 32768) for j in jobs)
        gib = estimate_gib(m, ctx)
        if gib is None:
            warn.append(f"{m}: не удалось оценить память")
        elif gib > MEM_LIMIT_GIB:
            warn.append(f"{m}: нужно {gib:.1f} GiB при контексте {ctx} (лимит {MEM_LIMIT_GIB}) — ночью будет пропущена")
            continue
        n_think = sum(1 for j in jobs if j.get("think"))
        n_plain = len(jobs) - n_think
        est = (n_plain * (median_sec(m, False) or 320) + n_think * (median_sec(m, True) or 1200)) / 3600
        total_h += est
        ok.append(f"{m}: {len(jobs)} прогонов, память ~{gib or '?'} GiB, ~{est:.1f} ч")
    print(f"\nОчередь {queue_path}: всего {len(tasks)}, уже готово {len(tasks) - len(todo)}, осталось {len(todo)}")
    print(f"Окружение: {env['backends']} | macOS {env['macos']}")
    for s in ok:
        print("  ✓", s)
    for s in warn:
        print("  ! ", s)
    for s in fail:
        print("  ✗", s)
    print(f"Оценка времени: ~{total_h:.1f} ч (грубо, уточняется по фактическим прогонам)")
    print("ГОТОВО К ЗАПУСКУ" if not fail else "ЕСТЬ БЛОКИРУЮЩИЕ ПРОБЛЕМЫ")
    return not fail


# ---------- основной цикл ----------

def parse_until(hhmm):
    """«07:00» → ближайший такой момент в будущем."""
    h, m = map(int, hhmm.split(":"))
    now_ = dt.datetime.now()
    at = now_.replace(hour=h, minute=m, second=0, microsecond=0)
    return at if at > now_ else at + dt.timedelta(days=1)


def deadline():
    """Срок читается из файла перед каждым прогоном — его можно сменить командой `until`."""
    try:
        return dt.datetime.fromisoformat(UNTIL_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def run(queue_path, until=None):
    for d in (RAW, FAILED, LOGS):
        d.mkdir(exist_ok=True)
    PAUSE.unlink(missing_ok=True)
    STOP.unlink(missing_ok=True)
    if until:
        UNTIL_FILE.write_text(until.isoformat(timespec="minutes"))
    else:
        UNTIL_FILE.unlink(missing_ok=True)
    _, tasks = load_queue(queue_path)
    env = environment(queue_path)
    st = Status(queue_path, tasks, env)
    models = downloaded_models()
    signal.signal(signal.SIGTERM, lambda *_: (st.event("получен сигнал остановки"), sys.exit(0)))

    # Очередь идёт строго по порядку файла и перечитывается перед каждым прогоном.
    # Модель всё равно перезагружается перед каждым прогоном, так что группировка по модели ничего не экономит.
    tried, fails, gib_by_model = set(), {}, {}
    while True:
        until = deadline()
        st.s["until"] = until.isoformat(timespec="minutes") if until else None
        if STOP.exists():
            st.event("остановка по команде stop")
            break
        if PAUSE.exists():
            sh(LMS, "unload", "--all", timeout=120)
            st.s["paused"] = True
            st.event("пауза: модель выгружена, жду команды resume")
            while PAUSE.exists() and not STOP.exists() and not (deadline() and dt.datetime.now() > deadline()):
                time.sleep(10)
            st.s["paused"] = False
            st.event("пауза снята")
            continue
        try:
            _, tasks = load_queue(queue_path)
        except Exception as e:  # noqa: BLE001 — опечатка в очереди посреди ночи не должна убить ночь
            st.event(f"очередь не читается ({e}) — работаю по прежней версии")
        todo = [x for x in pending(tasks) if x["name"] not in tried
                and x["job"]["model"] not in st.s["skipped_models"]
                and fails.get(job_key(x), 0) < FAILS_TO_SKIP_MODEL]
        st.s["total"], st.s["left"] = len(tasks), len(todo)
        if not todo:
            break
        t = todo[0]
        m, jkey = t["job"]["model"], job_key(t)
        if m not in models:
            models = downloaded_models()
        if m not in gib_by_model:
            ctx = max(x["job"].get("ctx", 32768) for x in todo if x["job"]["model"] == m)
            gib_by_model[m] = estimate_gib(m, ctx)
            if gib_by_model[m] and gib_by_model[m] > MEM_LIMIT_GIB:
                st.s["skipped_models"][m] = f"не влезает в память: {gib_by_model[m]:.1f} GiB"
                st.event(f"пропуск {m}: {gib_by_model[m]:.1f} GiB > {MEM_LIMIT_GIB}")
                continue
        # жёсткий срок: не начинать прогон, который не успеет закончиться
        secs = st.s["sec_by_job"].get(jkey) or st.s["sec_by_model"].get(m)
        need = (statistics.median(secs) if secs else 320 * (2 if t["job"].get("two_turn") else 1)) + 60
        if until and dt.datetime.now() + dt.timedelta(seconds=need) > until:
            st.event(f"срок {until:%H:%M}: следующий прогон (~{need / 60:.0f} мин) не успеет — останавливаюсь")
            break
        tried.add(t["name"])
        fails_in_row = fails.get(jkey, 0)
        st.s["current"] = t["name"]
        st.save()
        for attempt in range(1, ATTEMPTS + 1):
            try:
                if not ensure_server():
                    raise RuntimeError("сервер LM Studio не поднимается")
                # отпечаток окружения заново на каждый прогон: LM Studio может обновить движок посреди ночи
                meta = run_task(t, models.get(m, {}), environment(queue_path))
                if meta["env"]["backends"] != env["backends"]:
                    st.event(f"ВНИМАНИЕ: версия движка изменилась посреди ночи: {meta['env']['backends']}")
                st.s["done"] += 1
                st.s["truncated"] += meta["truncated"]
                st.s["parse_failed"] += not meta["parse_ok"]
                st.s["cache_contaminated"] += meta["cache_contaminated"]
                st.s["sec_by_model"].setdefault(m, []).append(meta["sec"])
                st.s["sec_by_job"].setdefault(jkey, []).append(meta["sec"] + meta["load_sec"])
                log(f"✓ {t['name']} {meta['sec']}с (+загрузка {meta['load_sec']}с), "
                    f"{meta['prompt_tokens']}+{meta['completion_tokens']} ток, кэш={meta['cache_cached_tokens']}, "
                    f"обрезан={meta['truncated']} json={meta['parse_ok']} батарея={meta['battery_pct']}%")
                if meta["cache_contaminated"] and t["job"].get("fresh_load", True):
                    st.event(f"ВНИМАНИЕ {t['name']}: сервер взял {meta['cache_cached_tokens']} токенов из кэша")
                # ответ без JSON или обрезанный — не сбой, но если так подряд, вариант жжёт ночь впустую
                if meta["parse_ok"] and not meta["truncated"]:
                    fails_in_row = 0
                else:
                    fails_in_row += 1
                    st.event(f"{t['name']}: ответ непригоден (json={meta['parse_ok']}, обрезан={meta['truncated']}), "
                             f"подряд {fails_in_row}")
                if meta["battery_pct"] is not None and meta["battery_pct"] < LOW_BATTERY_PCT:
                    st.event(f"батарея {meta['battery_pct']}% — зарядка не справляется, жду зарядки до 50%")
                    sh(LMS, "unload", "--all", timeout=120)
                    while (power()["battery_pct"] or 100) < 50 and not (deadline() and dt.datetime.now() > deadline()):
                        time.sleep(300)
                    st.event(f"батарея {power()['battery_pct']}% — продолжаю")
                grew = (meta["swap_after_mb"] or 0) - (meta["swap_before_mb"] or 0)
                if grew > SWAP_GROWTH_MB or (meta["mem_free_pct_after"] or 100) < MIN_FREE_PCT:
                    raise MemoryError(f"своп +{grew:.0f} МБ, свободно {meta['mem_free_pct_after']}%")
                break
            except MemoryError as e:
                st.s["skipped_models"][m] = f"нехватка памяти: {e}"
                st.event(f"{m}: {e} — пропускаю эту модель до конца ночи")
                sh(LMS, "unload", "--all", timeout=120)
                break
            except Exception as e:  # noqa: BLE001 — ночью любая ошибка должна стать записью, а не падением
                err = f"{type(e).__name__}: {str(e)[:300]}"
                st.s["last_error"] = f"{t['name']} попытка {attempt}: {err}"
                st.event(st.s["last_error"])
                (FAILED / f"{t['name']}.json").write_text(json.dumps(
                    {"name": t["name"], "attempt": attempt, "error": err, "at": now()}, ensure_ascii=False))
                if attempt < ATTEMPTS:
                    time.sleep(30 * attempt)
        else:
            st.s["failed"] += 1
            fails_in_row += 1
        fails[jkey] = fails_in_row
        if fails_in_row >= FAILS_TO_SKIP_MODEL:
            st.s.setdefault("skipped_jobs", {})[jkey] = "3 непригодных ответа или ошибки подряд"
            st.event(f"{jkey}: пропускаю оставшиеся прогоны этого варианта, остальные продолжаются")

    sh(LMS, "unload", "--all", timeout=120)
    st.s["current"] = None
    st.s["finished"] = now()
    st.event(f"ночь завершена: сделано {st.s['done']}, провалов {st.s['failed']}, "
             f"с кэшем {st.s['cache_contaminated']}")
    report = ROOT / "src" / "report.py"
    if report.exists():
        code, out = sh(sys.executable, str(report), timeout=1800)
        st.event("утренний отчёт собран" if code == 0 else f"отчёт не собрался: {out[-300:]}")


def start(queue_path, until=None):
    if LOCK.exists():
        pid = int(LOCK.read_text())
        try:
            os.kill(pid, 0)
            sys.exit(f"уже запущено (pid {pid}). Остановить: kill -- -{pid}")
        except ProcessLookupError:
            LOCK.unlink()
    if not check(queue_path):
        sys.exit("запуск отменён")
    LOGS.mkdir(exist_ok=True)
    logfile = LOGS / f"night_{dt.datetime.now():%Y-%m-%d_%H%M}.log"
    # caffeinate -i: не засыпать от бездействия, -s: не засыпать на зарядке. Экран гаснуть может.
    p = subprocess.Popen(["caffeinate", "-is", sys.executable, __file__, "run", str(queue_path)]
                         + (["--until", until] if until else []),
                         stdout=open(logfile, "w"), stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
    LOCK.write_text(str(p.pid))
    print(f"\nЗапущено в фоне (pid {p.pid}). Терминал можно закрыть.\n"
          f"  лог:    {logfile}\n  пульс:  {STATUS}\n  стоп:   kill -- -{p.pid}")


def status():
    if not STATUS.exists():
        sys.exit("ещё не запускалось")
    s = json.loads(STATUS.read_text())
    print(json.dumps({k: v for k, v in s.items() if k not in ("events", "env")}, ensure_ascii=False, indent=1))
    print("\nпоследние события:", *s["events"][-10:], sep="\n  ")
    print("\nпроцесс:", "работает" if LOCK.exists() else "не запущен",
          "| ПАУЗА" if PAUSE.exists() else "", "| срок", deadline() or "—")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    until_arg = sys.argv[sys.argv.index("--until") + 1] if "--until" in sys.argv else None
    if cmd == "start":
        start(sys.argv[2], until_arg)
    elif cmd == "run":
        try:
            run(sys.argv[2], parse_until(until_arg) if until_arg else None)
        finally:
            LOCK.unlink(missing_ok=True)
    elif cmd == "check":
        sys.exit(0 if check(sys.argv[2]) else 1)
    elif cmd == "status":
        status()
    elif cmd == "pause":
        PAUSE.touch()
        print("Пауза начнётся после текущего прогона (до ~10 мин): модель выгрузится, память освободится.\n"
              "Продолжить: night.py resume")
    elif cmd == "resume":
        PAUSE.unlink(missing_ok=True)
        print("Пауза снята — ночь продолжится в течение 10 секунд.")
    elif cmd == "stop":
        STOP.touch()
        print("Остановка после текущего прогона (до ~10 мин), затем соберётся отчёт.\n"
              "Остановить сразу: kill -- -$(cat logs/night.lock)")
    elif cmd == "until":
        at = parse_until(sys.argv[2])
        UNTIL_FILE.write_text(at.isoformat(timespec="minutes"))
        print(f"Новый срок: {at:%d.%m %H:%M}")
    else:
        sys.exit(__doc__)
