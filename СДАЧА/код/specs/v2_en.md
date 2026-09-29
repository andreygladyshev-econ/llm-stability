# Vector specification: 24 indicators of a bank report

Defines what is extracted from each quarterly bank report and the rules for assigning scores.
The report itself is in Russian. Indicator names are given in English with the Russian term in brackets.

---

## Main rule

A score reflects **the impact of the described fact on the investment attractiveness of the bank's shares** at the
moment the report is published.

| Score | When |
|:-:|---|
| **+2** | strong improvement: change above 15% year on year (for percentage indicators — above 1.5 p.p.) |
| **+1** | moderate improvement: from 3% to 15% (for percentage indicators — from 0.3 to 1.5 p.p.) |
| **0** | see the closed list below |
| **−1** | moderate deterioration |
| **−2** | strong deterioration |

Some indicators below have their own thresholds — those take priority over this table.

### The trap to remember

**The sign is determined by the impact on the shareholder, not by the direction of the number.**

- Loan loss provisions **grew** → score is **negative**. The number went up, but it is bad for the shareholder:
  the bank expects credit losses.
- Cost of risk **fell** → score is **positive**. The number went down, and that is good: portfolio quality improved.
- Cost-to-income ratio **decreased** → **plus**. The bank became more efficient.

For each indicator the direction is stated explicitly in the "plus means" column. Follow it, not intuition.

---

## Which period and which number to take

A report often contains several numbers for one indicator. Take exactly one, by these rules.

1. Profitability and efficiency indicators (net profit, EPS, ROE, net interest and fee income, NIM, operating
   expenses, cost-to-income ratio, cost of risk, provisions): take the value **for the latest quarter compared with
   the same quarter of the previous year**. In an annual report the latest quarter is Q4. If there is no quarterly
   year-on-year number, take the cumulative period year on year.
2. Loan portfolios, customer funds, book value per share, number of clients and users: **change since the start of
   the year**; if a figure excluding currency revaluation is given ("без учета валютной переоценки"), take it. If only
   the quarterly change is given, take that.
3. Thresholds apply to the chosen number as is, without annualising.
4. Numbers for other periods from the same report are not used, even if they are more striking.

---

## When the score is zero

This is a closed list. Zero is assigned **only** in four cases:

**(a) The indicator is not mentioned** — neither in the text nor in the tables.

**(b) A level is given, but no comparison with a previous period.**
Example: "return on equity was 24.0%" — no "year on year" and no base. There is an exception, see the next section.

**(c) The change is below the significance threshold** — below 3% for money amounts or 0.3 p.p. for ratios (for
indicators with their own thresholds — their lower threshold). This is noise, not a signal.

**(d) The report says explicitly there is no change** — "remained at last year's level", "unchanged", "stable".

**In all other cases the score is not zero.**

---

## "State" versus "flow"

Indicators fall into two classes, and rule (b) applies to them differently.

**Flow** — a value for a period, meaningful only compared with a base: profit, income, expenses, portfolio growth.
No comparison with a previous period → zero.

**State** — a characteristic at the reporting date, meaningful on its own: capital adequacy, cost of risk,
cost-to-income ratio, dividend stance, market share.

For state indicators, holding a level is a signal in itself, but **only if the report text contains an explicit
evaluative phrase next to the indicator**: "above the minimum" («выше минимума»), "comfortable level» («комфортный»),
"below target" («ниже целевого»), "within target", "under pressure" («под давлением»), etc. This phrase must be part of
the quote. If the report has no such phrase, score the change by the thresholds and do not decide on your own that the
level is comfortable.

| Wording in the report | Score | Why |
|---|:-:|---|
| "capital remains above the regulatory minimum" | **+1** | there is a safety margin |
| "adequacy decreased but remains comfortable" | **+1** | the final state named in the report outweighs the direction |
| "cost-to-income ratio is below the target level" | **+1** | costs under control |
| "cost of risk is held at a low level" | **+1** | asset quality is good |
| "capital under pressure", "below target level" | **−1** | the state is bad |
| "adequacy decreased by 1.2 p.p." (no evaluative phrase) | **by thresholds** | the text gives no assessment of the level |

---

## Vector composition

### Block 1. Profitability — 4 indicators

| id | Indicator | Plus means | Type |
|---|---|---|:-:|
| net_profit | Net profit (чистая прибыль) of the Group | growth year on year | flow |
| eps | Earnings per share (прибыль на акцию) | growth year on year | flow |
| roe | Return on equity (рентабельность капитала) | growth, in p.p. | flow |
| guidance | Management guidance (прогноз менеджмента): profit and ROE targets for the year | raised or confirmed | flow |

*EPS* may diverge from net profit if the bank bought back shares. A sign divergence is a signal in itself — note it.

*Guidance:* the **change** of guidance is scored, not its level. Target confirmed → +1. Withdrawn or lowered → −2.
No guidance → 0.

### Block 2. Income — 3 indicators

| id | Indicator | Plus means | Type |
|---|---|---|:-:|
| nii | Net interest income (чистые процентные доходы) | growth year on year | flow |
| fee_income | Net fee and commission income (чистые комиссионные доходы) | growth year on year | flow |
| nim | Net interest margin (чистая процентная маржа) | growth, in p.p. | flow |

### Block 3. Efficiency — 2 indicators

| id | Indicator | Plus means | Type |
|---|---|---|:-:|
| opex | Operating expenses (операционные расходы) | growing **slower** than operating income | flow |
| cir | Cost-to-income ratio (отношение расходов к доходам) | decrease | state |

*Operating expenses:* compare year-on-year growth of operating expenses with year-on-year growth of **operating income
before provisions** («операционный доход до резервов») for the same period — not with interest or fee income separately.
Gap = operating income growth − operating expense growth, in p.p.
- gap from −1 to +1 p.p. — **0**;
- above +1 up to +10 p.p. — **+1**; above +10 p.p. — **+2**;
- below −1 down to −10 p.p. — **−1**; below −10 p.p. — **−2**.

If operating income before provisions is not in the text — **0**, and say in the reasoning that there is nothing to
compare with.

### Block 4. Risk and asset quality — 3 indicators

| id | Indicator | Plus means | Type |
|---|---|---|:-:|
| cor | Cost of risk (стоимость риска) | decrease | state |
| provisions | Provision charges (расходы на резервы) | decrease | flow |
| portfolio_quality | Portfolio quality (качество портфеля): share of problem loans, quality of new lending | improvement | state |

*Cost of risk* is around 1%, the general thresholds do not apply. Change below 0.2 p.p. — **0**; from 0.2 to 0.5 p.p.
inclusive — **±1**; above 0.5 p.p. — **±2**. Growth is minus, decrease is plus. 100 basis points (бп) = 1 p.p.

The whole block is about how confident the bank is that loans will be repaid. Growing provisions mean the bank
expects losses. For the shareholder this is a minus, even if profit grew in the same quarter.

### Block 5. Balance sheet and business volumes — 4 indicators

| id | Indicator | Plus means | Type |
|---|---|---|:-:|
| corporate_loans | Corporate loan portfolio (корпоративный кредитный портфель) | growth | flow |
| retail_loans | Retail loan portfolio (розничный кредитный портфель) | growth | flow |
| customer_funds | Customer funds (средства клиентов) | growth | flow |
| market_share | Market share (доля рынка) by segments | share growing in most segments | state |

*Portfolios:* if the report gives a figure excluding currency revaluation, take exactly that. Portfolio growth caused
by a stronger dollar does not mean the bank lent more.

*Market share:* collect all reported changes of the bank's share by segment for the report period (retail lending,
corporate lending, SME, mortgages, credit cards, customer funds, etc.).
**+1** — more segments growing than falling; **−1** — more falling than growing; **0** — equal, or no changes of
share are reported. A share level without dynamics is not counted. ±2 is not used. Quote the sentence about the segment
with the largest change.

### Block 6. Capital and shareholder return — 3 indicators

| id | Indicator | Plus means | Type |
|---|---|---|:-:|
| capital_adequacy | Group capital adequacy ratio (Н20.0 or equivalent) (достаточность капитала) | growth, in p.p. | state |
| dividends | Dividends (дивиденды): announcement, size, willingness to pay | higher payout or confirmation | state |
| book_value_per_share | Book value per share (балансовая стоимость на акцию) | growth | flow |

*Dividends:* "record dividends" («рекордные») → +2. "We are returning to considering a payout" → +1. Refusal or
postponement → −2. Not mentioned → 0.

For a shareholder this indicator is often more important than profit: profit can stay in the bank, dividends arrive
in the account.

### Block 7. Client base and development — 3 indicators

| id | Indicator | Plus means | Type |
|---|---|---|:-:|
| active_clients | Active clients (активные клиенты): individuals and companies | growth | flow |
| digital_metrics | Digital metrics: MAU of the bank's main retail mobile app | growth | flow |
| tech_development | Technology development: AI, new platforms and services | concrete, implemented | flow |

*Digital metrics:* only **MAU (monthly active users) of the bank's main retail mobile app** is scored; if MAU
is not given — DAU of that app. Single sign-on ID, subscriptions, business apps, partner services and individual features are
not counted. The number of bank clients belongs to active_clients.

**Be careful with technology development.** This is the most promotional part of the report, written to impress.
Give +1 **only for a measurable fact**: a launch with a date, a number of clients, a share of processes.

- "The flagship model is already deeply integrated into the teams' daily work" → **0**. Nice, but nothing to verify.
- "The number of daily users grew by 6.8% over the quarter to 1.5 million" → **+1**. A concrete measurable statement.

Score the presence of a verifiable fact, not the tone.

### Block 8. External environment and tone — 2 indicators

| id | Indicator | Plus means | Type |
|---|---|---|:-:|
| external_conditions | External conditions: key rate, sanctions, regulation, demand | improving conditions | state |
| ceo_tone | CEO tone: the quote of the Chairman at the start of the report | confidence | state |

*External conditions* — score the environment, not the bank's result. A bank can perform well in bad conditions: that
is a plus for profit and a minus for the environment.

*Tone:* only the direct quote of the Chairman of the Board. Closed list of signs:
- **+2** — the word "record" («рекорд», «рекордный») or a concrete commitment for the future (target, plan, payout with a
  number);
- **+1** — facts of growth or improvement without reservations, including when the quote consists only of growth numbers;
- **0** — the quote has both plus signs and minus signs;
- **−1** — reservations, admitting problems: "despite" («несмотря на»), "pressure", "decline", "difficult conditions";
- **−2** — anti-crisis rhetoric: "the hardest" («сложнейший»), "crisis", "anti-crisis plan", "strictest austerity".

Two real examples:

> «По итогам работы в первом полугодии чистая прибыль выросла на 18,6% год к году при рентабельности капитала
> 24,2%» — **+1** (growth facts without reservations)

> «2022 год стал сложнейшим годом для нашей страны и банковского сектора… Мы реализовали антикризисный план:
> радикально пересмотрели приоритеты, ввели меры строжайшей экономии» — **−2** (anti-crisis rhetoric)

---

## How to quote

Every non-zero score needs a verbatim fragment of the report **in Russian, exactly as in the text**. Prefer a full
sentence from the text over a table row; quote a table row only if the number is not in the text. Copy the fragment
contiguously: no ellipses, no omissions, no comments in brackets. Do not translate the quote.

---

## If you find a contradiction in this specification

Report it. The specification was written by a person and certainly has places that can be read two ways.
