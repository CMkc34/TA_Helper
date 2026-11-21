# Strategie-Leitfaden für KI

Dieser Leitfaden beschreibt, wie eine Strategie für den Trading Helfer aussieht. Er ist so
geschrieben, dass eine KI (oder ein Mensch) daraus ohne Rückfrage eine Datei erzeugt, die der
Validator annimmt.

So kommt eine Strategie in die App:

1. Idee mit der KI ausarbeiten. Den [Prompt-Baustein](#prompt-baustein-für-den-ki-chat) am
   Ende dieses Leitfadens in den Chat kopieren und die Idee darunter beschreiben.
2. Die Antwort der KI (genau eine Python-Datei) im Tab **Strategien** über „Strategie
   hinzufügen“ einfügen oder als Datei wählen.
3. Der Validator prüft die Datei. Erst wenn alle sechs Stufen bestanden sind, wird sie im
   Plugin-Ordner gespeichert und erscheint in Backtest und Scanner.
4. Meldet der Validator einen Fehler, die Meldung (mit Zeilennummer) an die KI zurückgeben.

Alle Python-Blöcke in diesem Leitfaden sind vollständige Strategien. Ein Test
(`tests/unit/test_strategy_guide.py`) prüft, dass jeder davon den Validator besteht.

## Inhalt

1. [Wie die Engine eine Strategie ausführt](#wie-die-engine-eine-strategie-ausführt)
2. [Schnittstelle](#schnittstelle)
3. [Datenzugriff](#datenzugriff)
4. [Order-Arten und Positionsgröße](#order-arten-und-positionsgröße)
5. [Regeln](#regeln)
6. [Vorlage](#vorlage)
7. [Beispiel 1: Wochen-Trendfolge mit festem Risiko](#beispiel-1-wochen-trendfolge-mit-festem-risiko)
8. [Beispiel 2: Rücksetzer mit Kursziel und Zeit-Stop](#beispiel-2-rücksetzer-mit-kursziel-und-zeit-stop)
9. [Checkliste](#checkliste)
10. [Prompt-Baustein für den KI-Chat](#prompt-baustein-für-den-ki-chat)
11. [Validator-Meldungen verstehen](#validator-meldungen-verstehen)

## Wie die Engine eine Strategie ausführt

Eine Strategie handelt immer genau einen Wert in einem Zeitrahmen. Die Engine arbeitet so:

1. Sie lädt die bereinigten Tageskurse und rechnet sie in den Zeitrahmen um (`3D`, `1W`,
   `1M`). Die letzte, noch nicht abgeschlossene Kerze wird verworfen. Die Strategie sieht
   also **nur abgeschlossene Kerzen**.
2. Sie ruft **einmal** `indicators(bars, p)` mit der ganzen Kursreihe auf. Hier werden alle
   Indikatoren vektorisiert berechnet.
3. Für jede Kerze `t`:
   1. Zuerst führt der simulierte Broker die Orders aus, die auf der Vorkerze `t-1`
      abgegeben wurden (Market zum Open von `t`, Stop und Limit aus High und Low von `t`).
   2. Danach ruft die Engine `on_bar(ctx)` auf. Der Kontext enthält Kurse und Indikatoren
      bis einschließlich Close von `t`, die offene Position und das Kapital.
   3. Die zurückgegebenen Orders werden frühestens auf Kerze `t+1` ausgeführt.
4. **Warm-up:** `on_bar` wird erst ab der ersten Kerze aufgerufen, auf der *kein*
   Indikator NaN ist. Eine Spalte, die lange NaN bleibt, verzögert also den Start.
5. Offene Positionen am Ende bleiben offen und werden zum letzten Close bewertet.

Folgen für die Strategie:

- Eine Entscheidung fällt auf dem Close von `t`, gehandelt wird am Open von `t+1`. Nie mit
  einer Ausführung zum Close von `t` rechnen.
- Startkapital, Gebühren (1 USD je Order), Slippage und Bruchstücke stellt der Anwender in
  der App ein. Die Strategie kennt sie nicht und braucht sie nicht.
- Die Strategie entscheidet nur für `ctx.symbol`. Orders für andere Symbole lehnt die Engine
  ab.

## Schnittstelle

Eine Strategie ist eine Python-Datei mit (mindestens) einer Klasse, die von `Strategy` erbt.

### Pflichtfelder (Klassenattribute)

| Feld | Typ | Regel |
|---|---|---|
| `id` | `str` | Eindeutig, nur Kleinbuchstaben, Ziffern und `_`, beginnt mit einem Buchstaben, z. B. `"trend_ema_adx_risk"`. Nicht die ID einer eingebauten Strategie (`sma_cross`, `trend_donchian_breakout`, `mr_rsi2_trend`). |
| `name` | `str` | Anzeigename auf Deutsch. |
| `version` | `str` | SemVer, z. B. `"1.0.0"`. Bei jeder Regeländerung erhöhen. |
| `description` | `str` | Auf Deutsch: Einstieg, Ausstieg, gedachter Zeitrahmen, Long/Short. |
| `timeframes` | `ClassVar[list[Timeframe]]` | Nicht leer. Für diese App vor allem `Timeframe.D3` und `Timeframe.W1`, außerdem `Timeframe.MN1`. Der Validator testet jeden genannten Zeitrahmen. |
| `params` | `ClassVar[dict[str, Param]]` | Parameter, die Anwender und Optimierung ändern dürfen. Darf leer sein. |

### Parameter

`Param(default, min, max, step, description)` beschreibt einen Parameter:

- Der Typ folgt aus `default`: `int`, `float` oder `bool`. Ganzzahlige Perioden als `int`
  angeben (`Param(20, 5, 100, 1, …)`), Faktoren als `float` (`Param(3.0, 1.0, 6.0, 0.5, …)`).
- `min`, `max` und `step` legen das Optimierungsraster fest; `default` muss dazwischen liegen.
- Ein-/Aus-Schalter: `Param(True, description="Nur bei steigender SMA kaufen")` (ohne `min`, `max`, `step`).
- Auswahl aus festen Werten: `Param("ema", description="Art des Durchschnitts",
  choices=("sma", "ema"))`.
- Zugriff in `indicators` über `p.name` oder `p["name"]`. `on_bar` bekommt die Parameter
  nicht übergeben: in `indicators` mit `self._p = p` merken (siehe Vorlage) oder als
  konstante Indikator-Spalte ablegen.

### Methoden

| Methode | Pflicht | Aufgabe |
|---|---|---|
| `indicators(self, bars, p) -> pd.DataFrame` | nein | Alle Indikatoren vektorisiert über die ganze Reihe. Ergebnis hat dieselbe Zeilenzahl und denselben Index wie `bars`. Standard: keine Spalten. |
| `on_bar(self, ctx) -> list[Order]` | **ja** | Je Kerze die Orders zurückgeben; leere Liste heißt „nichts tun“. |
| `plot(self) -> list[PlotSpec]` | nein | Welche Indikator-Spalten der Chart zeigt. |
| `__init__(self)` | nein | Ohne weitere Pflicht-Argumente; die Engine legt die Strategie mit `Klasse()` an. |

`PlotSpec(column, kind="line", pane="price", label="", color="")`: `column` muss eine Spalte
aus `indicators` sein, `kind` ist `"line"`, `"marker"` oder `"histogram"`, `pane` ist
`"price"` (im Kurs-Chart) oder `"separate"` (eigenes Panel), `color` als `"#RRGGBB"`.

## Datenzugriff

### In `indicators(bars, p)`

`bars` ist ein `pd.DataFrame` mit einer Zeile je abgeschlossener Kerze des Zeitrahmens,
fortlaufendem Index `0 … n-1` und den Spalten `date`, `open`, `high`, `low`, `close`, `volume`.
Die Kurse sind um Splits und Dividenden bereinigt. Die Reihe beginnt vor dem Backtest-Start,
damit die Indikatoren anlaufen können.

Fertige Indikatoren liegen in `trading_helfer.indicators` (alle mit `df` als erstem Argument,
Ergebnis mit gleichem Index; die ersten Werte sind NaN, bis die Periode gefüllt ist):

| Modul | Funktion | Ergebnis |
|---|---|---|
| `trend` | `sma(df, period=20, source="close")`, `ema(…)`, `wma(…)` | Series |
| `trend` | `macd(df, fast=12, slow=26, signal=9)` | Spalten `macd`, `signal`, `hist` |
| `trend` | `adx(df, period=14)` | Spalten `adx`, `plus_di`, `minus_di` |
| `trend` | `supertrend(df, period=10, multiplier=3.0)` | Spalten `supertrend`, `direction` (`1.0`/`-1.0`) |
| `trend` | `ichimoku(df, tenkan=9, kijun=26, senkou=52)` | Spalten `tenkan`, `kijun`, `senkou_a`, `senkou_b` |
| `trend` | `parabolic_sar(df, acceleration=0.02, maximum=0.2)`, `atr_wilder(df, period=14)` | Series |
| `momentum` | `rsi(df, period=14)`, `roc(df, period=10)`, `williams_r(df, period=14)`, `cci(df, period=20)` | Series |
| `momentum` | `stochastic(df, k_period=14, k_smooth=3, d_period=3)` | Spalten `k`, `d` |
| `volatility` | `atr(df, period=14)` | Series |
| `volatility` | `bollinger(df, period=20, num_std=2.0)`, `keltner(df, period=20, multiplier=2.0, atr_period=10)` | Spalten `upper`, `middle`, `lower` |
| `volatility` | `donchian(df, period=20)` | Spalten `upper`, `middle`, `lower`; **die aktuelle Kerze zählt mit**, für Ausbrüche `.shift(1)` verwenden |
| `volatility` | `historical_volatility(df, period=20, annualization=252)` | Series (Dezimalzahl, `0.25` = 25 % p. a.) |
| `volume` | `obv(df)`, `vwap(df, period=20)`, `chaikin_money_flow(df, period=20)`, `volume_sma(df, period=20)` | Series |

Import zum Beispiel mit `from trading_helfer.indicators.trend import ema`. Eigene
Berechnungen mit `pandas` (`rolling`, `ewm`, `shift(1)`, `diff`, `pct_change`) und `numpy` sind
erlaubt, solange Zeile `t` nur Zeilen bis `t` verwendet. Die Relative-Stärke-Funktionen
brauchen einen Benchmark und sind in Strategien derzeit nicht nutzbar.

### In `on_bar(ctx)`: der `BarContext`

| Attribut oder Methode | Bedeutung |
|---|---|
| `ctx.symbol`, `ctx.date`, `ctx.timeframe` | Wert, Datum der Kerze `t` (`datetime.date`), Zeitrahmen |
| `ctx.position` | Offene Position oder `None`; Felder `side` (`Side.LONG`/`Side.SHORT`), `qty`, `entry_price`, `entry_date` |
| `ctx.cash`, `ctx.equity` | Freies Kapital, Gesamtkapital |
| `ctx.is_member` | `True`, wenn der Wert am Tag `t` zum Universum gehört. Neue Positionen nur dann eröffnen. |
| `ctx.value(name, ago=0)` | Wert einer Indikator- oder Kursspalte `ago` Kerzen vor `t` als `float`. Indikatoren haben Vorrang vor Kursen. Eine Zahl wird unverändert zurückgegeben. |
| `ctx.crossed_above(a, b)`, `ctx.crossed_below(a, b)` | Kreuzung auf Kerze `t` (`a[t-1] <= b[t-1]` und `a[t] > b[t]`); `a` und `b` sind Spaltennamen oder Zahlen. Bei NaN `False`. |
| `ctx.ind["sma"]`, `ctx.ind.sma` | Indikatorwert bei `t` |
| `ctx.ind.get("sma", ago=1)`, `ctx.ind.column("sma")`, `ctx.ind.row()` | Wert der Vorkerze, Historie bis `t` als Array, alle Werte bei `t` als `dict` |
| `ctx.bars.close`, `.open`, `.high`, `.low`, `.volume`, `.dates` | Historie bis einschließlich `t` als schreibgeschütztes `numpy`-Array; letzter Wert = Kerze `t` |
| `ctx.bars.get("close", ago=2)`, `len(ctx.bars)`, `ctx.bars.to_frame()` | Einzelwert, Anzahl Kerzen bis `t`, veränderbare Kopie als DataFrame |
| `ctx.buy(…)`, `ctx.sell_short(…)`, `ctx.exit(…)` | Order-Helfer, siehe nächster Abschnitt |

Ein Zugriff auf Kerzen nach `t` ist technisch nicht möglich: `ago` darf nicht negativ sein, und
die Arrays enden bei `t`. Ein solcher Versuch bricht den Backtest mit einem Lookahead-Fehler ab.

## Order-Arten und Positionsgröße

`on_bar` gibt eine Liste von `Order`s zurück. Am einfachsten über die Helfer des Kontexts:

| Helfer | Wirkung |
|---|---|
| `ctx.buy(qty=None, *, sizing=None, type=OrderType.MARKET, stop_price=None, limit_price=None, tag="")` | Long eröffnen oder aufstocken. Gibt **eine** `Order` zurück. |
| `ctx.sell_short(…)` | Short eröffnen oder aufstocken (gleiche Argumente). Gibt **eine** `Order` zurück. |
| `ctx.exit(qty=None, *, type=OrderType.MARKET, stop_price=None, limit_price=None, trail_pct=None, trail_atr=None, tag="")` | Offene Position schließen (ganz oder `qty` Stück). Gibt eine **Liste** zurück, leer ohne Position. |

Darum: `return [ctx.buy(…)]`, aber `return ctx.exit(…)`. Mehrere Ausstiege mit
`[*ctx.exit(…), *ctx.exit(…)]` zusammenfassen.

### Order-Typen (`OrderType` aus `trading_helfer.core.types`)

| Typ | Pflichtangabe | Ausführung |
|---|---|---|
| `MARKET` | – | Zum Open der nächsten Kerze. Gilt nur für diese eine Kerze. |
| `STOP` | `stop_price` | Kauf-Stop, wenn das High den Stop erreicht; Verkaufs-Stop, wenn das Low ihn erreicht. Bei einer Lücke zum Open. |
| `LIMIT` | `limit_price` | Kauf-Limit, wenn das Low das Limit erreicht; Verkaufs-Limit, wenn das High es erreicht. Höchstens zum Limitkurs. |
| `TRAILING_STOP` | genau eins von `trail_pct` (Prozent) oder `trail_atr` (Vielfaches der ATR(14)) | Wie Stop; der Stop wird nach jedem Close nachgezogen und bewegt sich nur in Richtung des Traders. Nur über `ctx.exit` oder `Order` setzbar. |

Wichtig:

- **Stop-, Limit- und Trailing-Orders bleiben offen**, bis sie ausgeführt werden. Eine
  Strategie kann sie nicht streichen. Darum Schutz-Stops und Kursziele **einmal** setzen, am
  besten auf der Kerze, auf der der Einstieg ausgeführt wurde (`ctx.position.entry_date ==
  ctx.date`), und nicht auf jeder Kerze neu.
- Wird eine Position geschlossen, streicht die Engine alle übrigen Ausstiegs-Orders dieser
  Position. Ein Market-Ausstieg räumt also offene Stops und Limits auf.
- Reihenfolge innerhalb einer Kerze: Market, dann Stops (inkl. Trailing), dann Limits.
  Erreicht eine Kerze Stop und Kursziel, gilt der Stop.
- Ausstiegs-Stops und -Limits dürfen schon vor dem Einstieg abgegeben werden; sie warten, bis
  die Position offen ist (siehe Beispiel 1). Ein Market-Ausstieg ohne Position wird verworfen.
- Eröffnende Stop- oder Limit-Orders bleiben ebenfalls offen. Für den Einstieg darum
  `MARKET` verwenden, außer die Strategie setzt eine solche Order bewusst nur einmal.
- Long und Short gleichzeitig im selben Wert geht nicht. Ohne ausdrücklichen Wunsch nur Long.

### Positionsgröße (`Sizing` aus `trading_helfer.strategies.base`)

Eröffnende Orders brauchen `qty` (Stückzahl, Bruchstücke erlaubt) **oder** `sizing`:

| Regel | Bedeutung |
|---|---|
| `Sizing.percent_equity(100)` | Prozent des aktuellen Kapitals (0 < Wert ≤ 100). Standard für Strategien mit einer Position. |
| `Sizing.fixed_amount(500)` | Fester Betrag in Kontowährung. |
| `Sizing.risk(1.0)` | So viele Stück, dass ein Treffer des Schutz-Stops 1 % des Kapitals kostet. Braucht einen offenen Ausstiegs-Stop, der **in derselben Liste** wie der Einstieg abgegeben wird (siehe Beispiel 1). |
| `Sizing.volatility_target(15)` | Positionsgröße für eine Ziel-Volatilität von 15 % p. a. (aus der historischen Volatilität über 20 Kerzen). |

Übersteigt eine Order das verfügbare Kapital, wird sie verworfen und im Protokoll vermerkt.
`tag="…"` beschreibt den Grund des Signals; der Text erscheint in Trade-Liste und Chart.

## Regeln

### Kein Lookahead

Der Wert einer Kerze `t` (Indikator oder Order) darf nur von Kerzen bis einschließlich `t`
abhängen. Der Validator verändert die Kurse nach mehreren Stichtagen und prüft, dass sich davor
nichts ändert.

| Falsch | Richtig |
|---|---|
| `close.shift(-1)`, `diff(-1)`, `pct_change(-1)` | `close.shift(1)` (Vorkerze) |
| `rolling(20, center=True)` | `rolling(20)` |
| `close / close.iloc[-1]`, `close.max()`, `close.mean()` über die ganze Reihe | `close.rolling(n).max()`, `close.expanding().max()` |
| Normieren oder Skalieren mit Statistik der ganzen Reihe | Statistik über ein rollierendes Fenster |
| `bfill()`, `fillna(method="bfill")`, `interpolate()` | `ffill()` oder NaN stehen lassen |
| `donchian(bars, n)["upper"]` als Ausbruchsschwelle | `donchian(bars, n)["upper"].shift(1)` (sonst kann der Close das eigene Hoch nie übersteigen) |

### Nur abgeschlossene Kerzen

Jede Entscheidung beruht auf dem Close von `t`. Kein Signal „während der Kerze“, keine
Annahme, zum Close von `t` gehandelt zu haben. Wer innerhalb der nächsten Kerze auf ein Niveau
reagieren will, setzt eine Stop- oder Limit-Order.

### Erlaubte Importe

Nur diese Module dürfen importiert werden:

| Modul | Wofür |
|---|---|
| `math`, `numpy`, `pandas` | Rechnen |
| `trading_helfer.indicators` (mit Untermodulen) | Fertige Indikatoren |
| `trading_helfer.strategies.base` | `Strategy`, `BarContext`, `Order`, `OrderSide`, `Param`, `Params`, `PlotSpec`, `Sizing` |
| `trading_helfer.core.types` | `Timeframe`, `OrderType`, `Side` |
| `typing`, `__future__` | Typangaben (`ClassVar`) |

Alles andere ist verboten, auch `datetime`, `random`, `os`, `pathlib`, `requests`,
`dataclasses`. Daten außer den übergebenen Kursen gibt es nicht: keine Dateien, kein Netz,
keine Fundamentaldaten.

### Verbotene Namen und Zugriffe

- Die Builtins `open`, `eval`, `exec`, `compile`, `__import__`, `globals`, `locals`, `vars`,
  `getattr`, `setattr`, `delattr`, `breakpoint`, `input`, `exit`, `quit`, `help`, `memoryview`
  sind verboten, **auch als Variablenname**. Also nicht `open = bars["open"]`, sondern
  `open_ = bars["open"]` oder `bars["open"]` direkt. (`ctx.exit(…)` und `ctx.bars.open` sind als
  Attribute erlaubt.)
- Keine Datei- oder Netzwerkfunktionen von `numpy` und `pandas`: alles mit `read_…` und
  `to_csv`, `to_json`, `to_pickle` usw., außerdem `np.load`, `np.save`, `.query(…)`, `.eval(…)`.
- Kein Zugriff auf Dunder-Attribute wie `__class__` oder `__dict__` (erlaubt sind
  `__init__`, `__name__`, `__doc__`).

### Determinismus

Gleiche Daten und Parameter müssen dasselbe Ergebnis liefern. Kein Zufall; falls doch nötig,
nur mit festem Seed (`np.random.default_rng(42)`). Keine Uhrzeit, kein Zustand, der von
einem Lauf zum nächsten überlebt. Zustand in `self` ist erlaubt, wird aber in `indicators`
neu gesetzt.

### Stil

- Eine Datei, eine Strategie-Klasse. Code und Bezeichner auf Englisch, Texte (`name`,
  `description`, `tag`, Parameterbeschreibungen) auf Deutsch.
- Keine Endlosschleifen: Jede Prüfstufe hat ein Zeitlimit von 30 Sekunden. Vektorisiert
  rechnen statt Schleifen über alle Kerzen in `on_bar`.
- Keine Ausgaben mit `print`; Gründe gehören in `tag`.

## Vorlage

Die Vorlage liegt auch als Datei in [`plugins/_template.py`](../plugins/_template.py). Sie
besteht den Validator unverändert. Zum Anpassen kopieren, alle Stellen mit `ANPASSEN` ändern
und ohne führenden Unterstrich speichern.

```python
"""Vorlage für eine eigene Strategie (siehe docs/strategy_guide.md).

Datei kopieren, ohne führenden Unterstrich speichern (z. B. ``plugins/meine_strategie.py``),
alle Stellen mit ``ANPASSEN`` ändern und im Strategien-Tab prüfen lassen. Dateien, deren
Name mit ``_`` beginnt, lädt die App nicht.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from trading_helfer.core.types import OrderType, Timeframe
from trading_helfer.indicators.trend import ema
from trading_helfer.indicators.volatility import atr
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Param,
    Params,
    PlotSpec,
    Sizing,
    Strategy,
)


class MeineStrategie(Strategy):  # ANPASSEN: Klassenname
    """ANPASSEN: Idee der Strategie in ein bis zwei Sätzen."""

    # Pflichtfelder ---------------------------------------------------------
    id = "meine_strategie"  # ANPASSEN: eindeutig, nur a-z, 0-9 und _
    name = "Meine Strategie"  # ANPASSEN: Anzeigename auf Deutsch
    version = "1.0.0"  # SemVer; bei jeder Regeländerung erhöhen
    description = (  # ANPASSEN: Einstieg, Ausstieg, Zeitrahmen, Long/Short
        "Kauf, wenn der Close über die EMA steigt; Ausstieg über einen Trailing Stop "
        "im Abstand von 3 × ATR(14). Für 3-Tages- und Wochenkerzen. Nur Long."
    )
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D3, Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        # Param(Standard, Minimum, Maximum, Schrittweite, Beschreibung)
        "ema_period": Param(30, 5, 100, 5, "Periode der Trend-EMA"),
        "atr_mult": Param(3.0, 1.0, 6.0, 0.5, "Abstand des Trailing Stops in ATR(14)"),
    }

    def __init__(self) -> None:
        """Legt die Strategie mit Standardparametern an."""
        self._p = self.make_params()

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        """Alle Indikatoren vektorisiert; Zeile t darf nur Kerzen bis t verwenden."""
        self._p = p  # Parameter für on_bar merken
        return pd.DataFrame(
            {
                "ema": ema(bars, p.ema_period),
                "atr": atr(bars, 14),
            },
            index=bars.index,
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Entscheidung auf dem Close der Kerze t; Ausführung am Open von t+1."""
        position = ctx.position
        if position is None:
            # Einstieg: nur wenn der Wert am Tag t zum Universum gehört.
            if ctx.is_member and ctx.crossed_above("close", "ema"):
                return [ctx.buy(sizing=Sizing.percent_equity(100), tag="Close über EMA")]
            return []
        if position.entry_date == ctx.date:
            # Kerze, auf der der Einstieg ausgeführt wurde: Schutz einmalig setzen.
            return ctx.exit(
                type=OrderType.TRAILING_STOP,
                trail_atr=float(self._p.atr_mult),
                tag="ATR-Trailing-Stop",
            )
        return []

    def plot(self) -> list[PlotSpec]:
        """Linien im Chart; jede Spalte muss aus indicators() stammen."""
        return [
            PlotSpec("ema", label="EMA", color="#1f77b4"),
            PlotSpec("atr", pane="separate", label="ATR(14)", color="#7f7f7f"),
        ]
```

## Beispiel 1: Wochen-Trendfolge mit festem Risiko

Zeigt: mehrere Indikatoren, Parameter in `on_bar`, Einstieg mit `Sizing.risk` und einem
Schutz-Stop, der zusammen mit dem Kauf abgegeben wird, Ausstieg per Market-Order, die den
offenen Stop aufräumt.

```python
"""Beispiel 1: Wochen-Trendfolge mit festem Risiko je Trade."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from trading_helfer.core.types import OrderType, Timeframe
from trading_helfer.indicators.trend import adx, ema
from trading_helfer.indicators.volatility import atr
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    OrderSide,
    Param,
    Params,
    PlotSpec,
    Sizing,
    Strategy,
)


class TrendRisiko(Strategy):
    """Kauft, wenn der Close bei starkem Trend über die EMA kreuzt; riskiert 1 % je Trade."""

    id = "trend_ema_adx_risk"
    name = "EMA-Trend mit Risiko-Stop"
    version = "1.0.0"
    description = (
        "Trendfolge: Kauf, wenn der Close über die EMA(30) kreuzt und der ADX(14) über 20 "
        "liegt. Schutz-Stop 2 × ATR(14) unter dem Einstiegssignal, Positionsgröße so, dass "
        "ein Stop-Treffer 1 % des Kapitals kostet. Ausstieg, wenn der Close unter die EMA "
        "fällt. Für Wochen- und 3-Tages-Kerzen. Nur Long."
    )
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1, Timeframe.D3]
    params: ClassVar[dict[str, Param]] = {
        "ema_period": Param(30, 10, 60, 5, "Periode der Trend-EMA"),
        "adx_min": Param(20.0, 10.0, 40.0, 5.0, "Mindeststärke des Trends (ADX)"),
        "stop_atr": Param(2.0, 1.0, 5.0, 0.5, "Abstand des Schutz-Stops in ATR(14)"),
        "risk_pct": Param(1.0, 0.25, 3.0, 0.25, "Risiko je Trade in % des Kapitals"),
    }

    def __init__(self) -> None:
        """Legt die Strategie mit Standardparametern an."""
        self._p = self.make_params()

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        """EMA, ADX und ATR; alle Werte bei t nutzen nur Kerzen bis t."""
        self._p = p
        return pd.DataFrame(
            {
                "ema": ema(bars, p.ema_period),
                "adx": adx(bars, 14)["adx"],
                "atr": atr(bars, 14),
            },
            index=bars.index,
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Einstieg mit Schutz-Stop in einem Schritt, Ausstieg bei Bruch der EMA."""
        p = self._p
        if ctx.position is None:
            trend = ctx.value("adx") > p.adx_min
            if ctx.is_member and trend and ctx.crossed_above("close", "ema"):
                stop = ctx.value("close") - p.stop_atr * ctx.value("atr")
                if stop <= 0:
                    return []
                # Der Schutz-Stop wird zusammen mit dem Kauf abgegeben. Er wartet, bis
                # die Position offen ist; Sizing.risk braucht ihn zur Größenberechnung.
                return [
                    ctx.buy(sizing=Sizing.risk(p.risk_pct), tag="EMA-Kreuzung"),
                    Order(
                        ctx.symbol,
                        OrderSide.SELL,
                        OrderType.STOP,
                        stop_price=stop,
                        tag="Schutz-Stop",
                    ),
                ]
            return []
        if ctx.crossed_below("close", "ema"):
            # Market-Ausstieg; offene Stops dieser Position streicht die Engine danach.
            return ctx.exit(tag="Close unter EMA")
        return []

    def plot(self) -> list[PlotSpec]:
        """EMA im Kurs-Chart, ADX in eigenem Panel."""
        return [
            PlotSpec("ema", label="EMA", color="#1f77b4"),
            PlotSpec("adx", pane="separate", label="ADX(14)", color="#9467bd"),
        ]
```

## Beispiel 2: Rücksetzer mit Kursziel und Zeit-Stop

Zeigt: Mean-Reversion auf 3-Tages-Kerzen, Kursziel (`LIMIT`) und Schutz-Stop (`STOP`) einmalig
auf der Einstiegskerze, Zeit-Stop über die Anzahl Kerzen seit dem Einstieg.

```python
"""Beispiel 2: Rücksetzer im Aufwärtstrend mit Kursziel und Zeit-Stop (3-Tages-Kerzen)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from trading_helfer.core.types import OrderType, Timeframe
from trading_helfer.indicators.trend import sma
from trading_helfer.indicators.volatility import bollinger
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Param,
    Params,
    PlotSpec,
    Sizing,
    Strategy,
)


class BollingerRuecksetzer(Strategy):
    """Kauft Schlusskurse unter dem unteren Bollinger-Band, solange der Trend steigt."""

    id = "mr_bollinger_pullback"
    name = "Bollinger-Rücksetzer im Aufwärtstrend"
    version = "1.0.0"
    description = (
        "Mean-Reversion: Kauf, wenn der Close unter dem unteren Bollinger-Band (20, 2) "
        "schließt und über der SMA 100 liegt. Ausstieg am Kursziel (Limit 6 % über dem "
        "Einstieg), am Schutz-Stop (8 % darunter), sobald der Close die Mittellinie "
        "erreicht oder spätestens nach 10 Kerzen. Für 3-Tages-Kerzen. Nur Long."
    )
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D3]
    params: ClassVar[dict[str, Param]] = {
        "bb_period": Param(20, 10, 40, 5, "Periode der Bollinger-Bänder"),
        "bb_std": Param(2.0, 1.5, 3.0, 0.25, "Bandabstand in Standardabweichungen"),
        "trend_period": Param(100, 50, 200, 10, "Periode der Trend-SMA"),
        "target_pct": Param(6.0, 2.0, 15.0, 1.0, "Kursziel in % über dem Einstieg"),
        "stop_pct": Param(8.0, 3.0, 15.0, 1.0, "Schutz-Stop in % unter dem Einstieg"),
        "max_bars": Param(10, 3, 30, 1, "Spätester Ausstieg nach so vielen Kerzen"),
    }

    def __init__(self) -> None:
        """Legt die Strategie mit Standardparametern an."""
        self._p = self.make_params()

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        """Bollinger-Bänder und Trend-SMA."""
        self._p = p
        bands = bollinger(bars, p.bb_period, p.bb_std)
        return pd.DataFrame(
            {
                "bb_lower": bands["lower"],
                "bb_middle": bands["middle"],
                "trend_sma": sma(bars, p.trend_period),
            },
            index=bars.index,
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Einstieg am Band, danach Ziel, Stop, Mittellinie oder Zeit."""
        p = self._p
        position = ctx.position
        close = ctx.value("close")
        if position is None:
            uptrend = close > ctx.value("trend_sma")
            if ctx.is_member and uptrend and close < ctx.value("bb_lower"):
                return [ctx.buy(sizing=Sizing.percent_equity(100), tag="Unter Bollinger-Band")]
            return []

        if position.entry_date == ctx.date:
            # Einstieg wurde am Open dieser Kerze ausgeführt: Ziel und Stop einmalig
            # setzen. Beide bleiben offen, bis einer ausgelöst oder die Position
            # anders geschlossen wird; dann streicht die Engine den anderen.
            entry = position.entry_price
            return [
                *ctx.exit(
                    type=OrderType.LIMIT,
                    limit_price=entry * (1 + p.target_pct / 100),
                    tag="Kursziel",
                ),
                *ctx.exit(
                    type=OrderType.STOP,
                    stop_price=entry * (1 - p.stop_pct / 100),
                    tag="Schutz-Stop",
                ),
            ]

        # Zeit-Stop: Kerzen seit dem Einstieg zählen (nur Daten bis t).
        held = int(np.count_nonzero(ctx.bars.dates >= np.datetime64(position.entry_date)))
        if held > p.max_bars:
            return ctx.exit(tag="Zeit-Stop")
        if close >= ctx.value("bb_middle"):
            return ctx.exit(tag="Mittellinie erreicht")
        return []

    def plot(self) -> list[PlotSpec]:
        """Unteres Band, Mittellinie und Trend-SMA im Kurs-Chart."""
        return [
            PlotSpec("bb_lower", label="Bollinger unten", color="#d62728"),
            PlotSpec("bb_middle", label="Bollinger Mitte", color="#7f7f7f"),
            PlotSpec("trend_sma", label="SMA Trend", color="#ff7f0e"),
        ]
```

## Checkliste

Vor dem Einfügen in die App:

- [ ] Genau eine Python-Datei mit einer Klasse, die von `Strategy` erbt.
- [ ] `id` eindeutig (`a-z`, `0-9`, `_`), `name` und `description` auf Deutsch, `version` als
      SemVer, `timeframes` nicht leer (für diese App `D3` und/oder `W1`).
- [ ] Jeder Parameter als `Param(default, min, max, step, beschreibung)`, `default` liegt
      zwischen `min` und `max`; Perioden als `int`.
- [ ] Nur erlaubte Importe; kein Builtin wie `open` oder `exit` als Variablenname.
- [ ] `indicators` liefert ein DataFrame mit `index=bars.index`; kein `shift(-n)`, kein
      `center=True`, keine Statistik über die ganze Reihe.
- [ ] Spalten, die lange NaN sind, verzögern den Start; Ausbruchsschwellen mit `shift(1)`.
- [ ] `on_bar` gibt immer eine Liste zurück: `[ctx.buy(…)]`, `ctx.exit(…)` oder `[]`.
- [ ] Neue Positionen nur, wenn `ctx.position is None` (oder bewusst aufstocken) und
      `ctx.is_member`.
- [ ] Stops und Kursziele nur einmal setzen (`position.entry_date == ctx.date` oder zusammen
      mit dem Einstieg), nicht auf jeder Kerze.
- [ ] `Sizing.risk` nur zusammen mit einem Ausstiegs-Stop in derselben Liste.
- [ ] Jede Spalte in `plot()` kommt aus `indicators`.
- [ ] Kein Zufall ohne Seed, keine Dateien, kein Netz, kein `print`.

## Prompt-Baustein für den KI-Chat

Den folgenden Block vollständig in den KI-Chat kopieren und die eigene Idee an der markierten
Stelle eintragen. Bei Fehlern den Bericht des Validators zurück in denselben Chat geben.

````text
Du schreibst eine Handelsstrategie als Python-Plugin für die Desktop-App „Trading Helfer“
(Backtesting von US-Aktien). Antworte mit genau EINER vollständigen Python-Datei in einem
Code-Block und danach höchstens fünf Sätzen Erklärung auf Deutsch.

Schnittstelle:
- Eine Klasse, die von trading_helfer.strategies.base.Strategy erbt.
- Klassenattribute: id (eindeutig, nur a-z, 0-9, _), name (Deutsch), version (SemVer),
  description (Deutsch: Einstieg, Ausstieg, Zeitrahmen, Long/Short),
  timeframes: ClassVar[list[Timeframe]] (Timeframe.D3 und/oder Timeframe.W1),
  params: ClassVar[dict[str, Param]] mit Param(default, min, max, step, "Beschreibung").
- indicators(self, bars, p) -> pd.DataFrame: einmal vektorisiert über die ganze Reihe.
  bars hat die Spalten date, open, high, low, close, volume (bereinigt), Index 0..n-1.
  Rückgabe mit index=bars.index. Parameter mit self._p = p für on_bar merken.
- on_bar(self, ctx) -> list[Order]: je abgeschlossener Kerze t; Orders werden ab dem Open
  von t+1 ausgeführt. on_bar läuft erst, wenn keine Indikator-Spalte mehr NaN ist.
  ctx.value("spalte", ago=0) -> float, ctx.crossed_above(a, b), ctx.crossed_below(a, b),
  ctx.position (None oder side, qty, entry_price, entry_date), ctx.date, ctx.is_member,
  ctx.cash, ctx.equity, ctx.bars.close (Array bis t), ctx.ind["spalte"].
- Orders: [ctx.buy(sizing=Sizing.percent_equity(100), tag="Grund")], ctx.exit(tag="Grund")
  (gibt schon eine Liste zurück), ctx.exit(type=OrderType.STOP, stop_price=…),
  ctx.exit(type=OrderType.LIMIT, limit_price=…),
  ctx.exit(type=OrderType.TRAILING_STOP, trail_atr=3.0) oder trail_pct=10.0.
  Sizing.risk(1.0) nur zusammen mit Order(ctx.symbol, OrderSide.SELL, OrderType.STOP,
  stop_price=…) in derselben Liste.
  Stop-, Limit- und Trailing-Orders bleiben offen, bis sie auslösen, und lassen sich nicht
  streichen: nur einmal setzen (auf der Kerze mit position.entry_date == ctx.date). Beim
  Schließen der Position streicht die Engine die übrigen Ausstiegs-Orders.
- plot(self) -> list[PlotSpec]: PlotSpec("spalte", pane="price" oder "separate",
  label="…", color="#RRGGBB"); nur Spalten aus indicators.

Fertige Indikatoren (erstes Argument df):
trading_helfer.indicators.trend: sma, ema, wma (period, source="close"), macd (Spalten
macd, signal, hist), adx (adx, plus_di, minus_di), supertrend (supertrend, direction),
ichimoku, parabolic_sar, atr_wilder.
trading_helfer.indicators.momentum: rsi, roc, williams_r, cci, stochastic (k, d).
trading_helfer.indicators.volatility: atr, bollinger / keltner / donchian (upper, middle,
lower; donchian enthält die aktuelle Kerze, für Ausbrüche .shift(1)),
historical_volatility.
trading_helfer.indicators.volume: obv, vwap, chaikin_money_flow, volume_sma.

Regeln (sonst lehnt der Validator die Datei ab):
1. Kein Lookahead: Zeile t darf nur Kerzen bis t verwenden. Kein shift(-n), kein
   center=True, kein bfill, keine Statistik über die ganze Reihe (max, mean, iloc[-1]).
2. Nur diese Importe: math, numpy, pandas, trading_helfer.indicators.*,
   trading_helfer.strategies.base, trading_helfer.core.types, typing, __future__.
3. Verboten, auch als Variablenname: open, eval, exec, compile, __import__, globals,
   locals, vars, getattr, setattr, delattr, breakpoint, input, exit, quit, help,
   memoryview. Keine Dateien, kein Netz (kein read_*, to_csv, np.load, .query, .eval),
   keine Dunder-Attribute wie __class__.
4. Deterministisch: kein Zufall ohne festen Seed, keine Uhrzeit, kein print.
5. Neue Positionen nur bei ctx.position is None und ctx.is_member. Nur Long, außer ich
   wünsche ausdrücklich Short.
6. Code und Bezeichner auf Englisch, alle Texte auf Deutsch.

Halte dich an diese Vorlage:

```python
"""Vorlage für eine eigene Strategie (siehe docs/strategy_guide.md).

Datei kopieren, ohne führenden Unterstrich speichern (z. B. ``plugins/meine_strategie.py``),
alle Stellen mit ``ANPASSEN`` ändern und im Strategien-Tab prüfen lassen. Dateien, deren
Name mit ``_`` beginnt, lädt die App nicht.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from trading_helfer.core.types import OrderType, Timeframe
from trading_helfer.indicators.trend import ema
from trading_helfer.indicators.volatility import atr
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Param,
    Params,
    PlotSpec,
    Sizing,
    Strategy,
)


class MeineStrategie(Strategy):  # ANPASSEN: Klassenname
    """ANPASSEN: Idee der Strategie in ein bis zwei Sätzen."""

    # Pflichtfelder ---------------------------------------------------------
    id = "meine_strategie"  # ANPASSEN: eindeutig, nur a-z, 0-9 und _
    name = "Meine Strategie"  # ANPASSEN: Anzeigename auf Deutsch
    version = "1.0.0"  # SemVer; bei jeder Regeländerung erhöhen
    description = (  # ANPASSEN: Einstieg, Ausstieg, Zeitrahmen, Long/Short
        "Kauf, wenn der Close über die EMA steigt; Ausstieg über einen Trailing Stop "
        "im Abstand von 3 × ATR(14). Für 3-Tages- und Wochenkerzen. Nur Long."
    )
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D3, Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        # Param(Standard, Minimum, Maximum, Schrittweite, Beschreibung)
        "ema_period": Param(30, 5, 100, 5, "Periode der Trend-EMA"),
        "atr_mult": Param(3.0, 1.0, 6.0, 0.5, "Abstand des Trailing Stops in ATR(14)"),
    }

    def __init__(self) -> None:
        """Legt die Strategie mit Standardparametern an."""
        self._p = self.make_params()

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        """Alle Indikatoren vektorisiert; Zeile t darf nur Kerzen bis t verwenden."""
        self._p = p  # Parameter für on_bar merken
        return pd.DataFrame(
            {
                "ema": ema(bars, p.ema_period),
                "atr": atr(bars, 14),
            },
            index=bars.index,
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Entscheidung auf dem Close der Kerze t; Ausführung am Open von t+1."""
        position = ctx.position
        if position is None:
            # Einstieg: nur wenn der Wert am Tag t zum Universum gehört.
            if ctx.is_member and ctx.crossed_above("close", "ema"):
                return [ctx.buy(sizing=Sizing.percent_equity(100), tag="Close über EMA")]
            return []
        if position.entry_date == ctx.date:
            # Kerze, auf der der Einstieg ausgeführt wurde: Schutz einmalig setzen.
            return ctx.exit(
                type=OrderType.TRAILING_STOP,
                trail_atr=float(self._p.atr_mult),
                tag="ATR-Trailing-Stop",
            )
        return []

    def plot(self) -> list[PlotSpec]:
        """Linien im Chart; jede Spalte muss aus indicators() stammen."""
        return [
            PlotSpec("ema", label="EMA", color="#1f77b4"),
            PlotSpec("atr", pane="separate", label="ATR(14)", color="#7f7f7f"),
        ]
```

Meine Strategie-Idee:
<HIER DIE IDEE BESCHREIBEN: Markt/Universum, Zeitrahmen 3D oder 1W, Einstieg, Ausstieg,
Stop, Positionsgröße>
````

## Validator-Meldungen verstehen

Der Validator prüft in sechs Stufen; scheitert eine, werden die folgenden übersprungen.

| Stufe | Typische Ursache | Abhilfe |
|---|---|---|
| Syntax | Unvollständiger Code, falsche Einrückung | Ganze Datei neu erzeugen lassen. |
| Sicherheit (Importe und Aufrufe) | Verbotener Import, `open` als Variable, `getattr`, `.query` | Siehe [Regeln](#regeln); Meldung nennt die Zeile. |
| Pflichtfelder und Typen | Feld fehlt, `version` kein SemVer, `default` außerhalb von `min`/`max`, falsche Signatur von `on_bar` oder `indicators` | Felder und Signaturen wie in der Vorlage. |
| Probelauf auf Testdaten | Ausnahme in `indicators` oder `on_bar`, `on_bar` gibt keine Liste zurück, `plot` nennt unbekannte Spalte | Meldung mit Datum und Zeile an die KI geben. |
| Determinismus | Zufall ohne Seed, Zustand aus einem früheren Lauf | Seed setzen, Zustand in `indicators` zurücksetzen. |
| Lookahead-Test | `shift(-n)`, `center=True`, Statistik über die ganze Reihe | Siehe [Kein Lookahead](#kein-lookahead). |

Die Testdaten sind synthetische Tageskurse 2000–2014 mit Auf- und Abwärtstrends. Dass eine
Strategie den Validator besteht, sagt nichts über ihre Qualität; das zeigt erst der Backtest.
