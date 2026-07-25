"use client";

import { ChangeEvent, useMemo, useState } from "react";

type Row = Record<string, string>;
type Profile = {
  name: string;
  type: "数值" | "文本";
  missing: number;
  unique: number;
  mean: number | null;
  min: number | null;
  max: number | null;
};
type Estimate = {
  term: string;
  coefficient: number;
  standardError: number;
  statistic: number;
};

const demoRows: Row[] = Array.from({ length: 36 }, (_, i) => {
  const treatment = i >= 18 ? 1 : 0;
  const age = 22 + ((i * 7) % 25);
  const education = 9 + ((i * 5) % 10);
  const experience = Math.max(0, age - education - 6);
  const noise = ((i * 13) % 9 - 4) * 0.07;
  const income =
    2.1 + 0.31 * treatment + 0.075 * education + 0.018 * experience + noise;
  return {
    income: income.toFixed(3),
    treatment: String(treatment),
    education: String(education),
    experience: String(experience),
    region: ["东部", "中部", "西部"][i % 3],
  };
});

function parseCSV(text: string): Row[] {
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const split = (line: string) => {
    const values: string[] = [];
    let current = "";
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        if (quoted && line[i + 1] === '"') {
          current += '"';
          i++;
        } else quoted = !quoted;
      } else if (char === "," && !quoted) {
        values.push(current.trim());
        current = "";
      } else current += char;
    }
    values.push(current.trim());
    return values;
  };
  const headers = split(lines[0]);
  return lines.slice(1).map((line) => {
    const values = split(line);
    return Object.fromEntries(headers.map((header, i) => [header, values[i] ?? ""]));
  });
}

function transpose(matrix: number[][]) {
  return matrix[0].map((_, column) => matrix.map((row) => row[column]));
}

function multiply(a: number[][], b: number[][]) {
  return a.map((row) =>
    b[0].map((_, column) =>
      row.reduce((sum, value, i) => sum + value * b[i][column], 0),
    ),
  );
}

function inverse(matrix: number[][]) {
  const n = matrix.length;
  const augmented = matrix.map((row, i) => [
    ...row,
    ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  ]);
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(augmented[j][i]) > Math.abs(augmented[pivot][i])) pivot = j;
    }
    [augmented[i], augmented[pivot]] = [augmented[pivot], augmented[i]];
    const divisor = augmented[i][i];
    if (Math.abs(divisor) < 1e-10) throw new Error("解释变量之间存在完全共线性");
    augmented[i] = augmented[i].map((value) => value / divisor);
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const factor = augmented[j][i];
      augmented[j] = augmented[j].map(
        (value, k) => value - factor * augmented[i][k],
      );
    }
  }
  return augmented.map((row) => row.slice(n));
}

function runOLS(rows: Row[], outcome: string, predictors: string[]) {
  const clean = rows
    .map((row) => ({
      y: Number(row[outcome]),
      x: [1, ...predictors.map((name) => Number(row[name]))],
    }))
    .filter(({ y, x }) => Number.isFinite(y) && x.every(Number.isFinite));
  if (clean.length <= predictors.length + 2) throw new Error("有效样本量不足");
  const x = clean.map((item) => item.x);
  const y = clean.map((item) => [item.y]);
  const xtxInverse = inverse(multiply(transpose(x), x));
  const beta = multiply(multiply(xtxInverse, transpose(x)), y).map((v) => v[0]);
  const residuals = clean.map((item, i) => item.y - multiply([item.x], beta.map((v) => [v]))[0][0]);
  const degrees = clean.length - beta.length;
  const sigma2 = residuals.reduce((sum, value) => sum + value * value, 0) / degrees;
  const standardErrors = xtxInverse.map((row, i) =>
    Math.sqrt(Math.max(0, sigma2 * row[i])),
  );
  const meanY = clean.reduce((sum, item) => sum + item.y, 0) / clean.length;
  const tss = clean.reduce((sum, item) => sum + (item.y - meanY) ** 2, 0);
  const rss = residuals.reduce((sum, value) => sum + value ** 2, 0);
  const terms = ["截距", ...predictors];
  return {
    estimates: terms.map((term, i) => ({
      term,
      coefficient: beta[i],
      standardError: standardErrors[i],
      statistic: beta[i] / standardErrors[i],
    })),
    n: clean.length,
    r2: 1 - rss / tss,
  };
}

export default function Home() {
  const [rows, setRows] = useState<Row[]>(demoRows);
  const [fileName, setFileName] = useState("内置演示数据");
  const [outcome, setOutcome] = useState("income");
  const [predictors, setPredictors] = useState<string[]>([
    "treatment",
    "education",
    "experience",
  ]);
  const [estimates, setEstimates] = useState<Estimate[]>([]);
  const [fit, setFit] = useState<{ n: number; r2: number } | null>(null);
  const [error, setError] = useState("");
  const [supportOpen, setSupportOpen] = useState(false);

  const columns = useMemo(() => Object.keys(rows[0] ?? {}), [rows]);
  const profiles = useMemo<Profile[]>(() => {
    return columns.map((name) => {
      const values = rows.map((row) => row[name]);
      const present = values.filter((value) => value !== "");
      const numeric = present.map(Number);
      const isNumeric = present.length > 0 && numeric.every(Number.isFinite);
      return {
        name,
        type: isNumeric ? "数值" : "文本",
        missing: values.length - present.length,
        unique: new Set(present).size,
        mean: isNumeric ? numeric.reduce((a, b) => a + b, 0) / numeric.length : null,
        min: isNumeric ? Math.min(...numeric) : null,
        max: isNumeric ? Math.max(...numeric) : null,
      };
    });
  }, [columns, rows]);
  const numericColumns = profiles.filter((item) => item.type === "数值").map((item) => item.name);
  const missingCount = profiles.reduce((sum, item) => sum + item.missing, 0);

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const parsed = parseCSV(String(reader.result));
      if (!parsed.length) {
        setError("没有读取到有效数据，请检查 CSV 文件");
        return;
      }
      const names = Object.keys(parsed[0]);
      const numeric = names.filter((name) =>
        parsed.every((row) => row[name] === "" || Number.isFinite(Number(row[name]))),
      );
      setRows(parsed);
      setFileName(file.name);
      setOutcome(numeric[0] ?? "");
      setPredictors(numeric.slice(1, 4));
      setEstimates([]);
      setFit(null);
      setError("");
    };
    reader.readAsText(file);
  }

  function estimate() {
    try {
      const result = runOLS(rows, outcome, predictors);
      setEstimates(result.estimates);
      setFit({ n: result.n, r2: result.r2 });
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "模型估计失败");
    }
  }

  function exportResult() {
    if (!estimates.length) return;
    const csv = [
      ["变量", "系数", "标准误", "t值"].join(","),
      ...estimates.map((item) =>
        [item.term, item.coefficient, item.standardError, item.statistic].join(","),
      ),
    ].join("\n");
    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "计量工坊_回归结果.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="计量工坊首页">
          <span className="brandMark">计</span>
          <span>计量工坊</span>
          <em>StatsPAI Lab</em>
        </a>
        <nav>
          <a href="#data">数据</a>
          <a href="#model">模型</a>
          <a href="#result">结果</a>
        </nav>
        <button className="supportButton" onClick={() => setSupportOpen(true)}>
          支持项目
        </button>
      </header>

      <section className="hero" id="top">
        <div className="heroCopy">
          <span className="eyebrow">LOCAL-FIRST ECONOMETRICS</span>
          <h1>让实证分析，<br /><i>少一点门槛。</i></h1>
          <p>
            在浏览器中完成数据体检与基准回归。文件只在你的设备内处理，
            不上传服务器。
          </p>
          <div className="heroActions">
            <label className="uploadPrimary">
              上传 CSV 数据
              <input type="file" accept=".csv,text/csv" onChange={handleFile} />
            </label>
            <a href="#model" className="textLink">使用演示数据 <span>→</span></a>
          </div>
        </div>
        <div className="heroPanel" aria-label="分析流程预览">
          <div className="panelTop">
            <span>分析工作台</span>
            <span className="live"><b /> 本地运行</span>
          </div>
          <div className="miniChart">
            {[42, 61, 48, 78, 66, 88, 72, 94].map((height, i) => (
              <span key={i} style={{ height: `${height}%` }} />
            ))}
          </div>
          <div className="metricRow">
            <div><span>样本量</span><strong>{rows.length}</strong></div>
            <div><span>变量数</span><strong>{columns.length}</strong></div>
            <div><span>缺失值</span><strong>{missingCount}</strong></div>
          </div>
          <div className="codeLine">
            <span>model</span>
            <code>{outcome || "y"} ~ {predictors.join(" + ") || "x"}</code>
          </div>
        </div>
      </section>

      <section className="workspace" id="data">
        <div className="sectionHeading">
          <div>
            <span className="step">01 / 数据体检</span>
            <h2>先读懂数据，再开始回归</h2>
          </div>
          <div className="filePill"><span>●</span>{fileName}</div>
        </div>
        <div className="profileGrid">
          {profiles.slice(0, 6).map((profile) => (
            <article className="profileCard" key={profile.name}>
              <div><strong>{profile.name}</strong><span>{profile.type}</span></div>
              <dl>
                <div><dt>缺失</dt><dd>{profile.missing}</dd></div>
                <div><dt>唯一值</dt><dd>{profile.unique}</dd></div>
                <div><dt>{profile.type === "数值" ? "均值" : "类型"}</dt>
                  <dd>{profile.mean === null ? "分类" : profile.mean.toFixed(2)}</dd></div>
              </dl>
              {profile.mean !== null && (
                <div className="range"><span style={{ width: `${Math.min(100, 24 + profile.unique * 4)}%` }} /></div>
              )}
            </article>
          ))}
        </div>
      </section>

      <section className="modelSection" id="model">
        <div className="sectionHeading light">
          <div>
            <span className="step">02 / 模型设定</span>
            <h2>用清晰的语言定义模型</h2>
          </div>
          <span className="methodBadge">OLS · 经典标准误</span>
        </div>
        <div className="modelBuilder">
          <div className="field">
            <label htmlFor="outcome">被解释变量 Y</label>
            <select id="outcome" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
              {numericColumns.map((column) => <option key={column}>{column}</option>)}
            </select>
          </div>
          <div className="formulaSymbol">=</div>
          <div className="field predictors">
            <label>解释变量 X（点击选择）</label>
            <div className="variableChoices">
              {numericColumns.filter((column) => column !== outcome).map((column) => (
                <button
                  key={column}
                  className={predictors.includes(column) ? "selected" : ""}
                  onClick={() => setPredictors((current) =>
                    current.includes(column)
                      ? current.filter((item) => item !== column)
                      : [...current, column],
                  )}
                >
                  {column}<span>{predictors.includes(column) ? "×" : "+"}</span>
                </button>
              ))}
            </div>
          </div>
          <button className="runButton" onClick={estimate}>运行回归 <span>↗</span></button>
        </div>
        {error && <p className="error">{error}</p>}
        <p className="modelNote">
          当前版本提供教学与初步探索用途的基准 OLS。正式研究需要进一步核查识别假设、
          聚类层级、固定效应及稳健性。
        </p>
      </section>

      <section className="resultSection" id="result">
        <div className="sectionHeading">
          <div>
            <span className="step">03 / 估计结果</span>
            <h2>{estimates.length ? "结果已就绪" : "运行模型后查看结果"}</h2>
          </div>
          {estimates.length > 0 && <button className="exportButton" onClick={exportResult}>导出 CSV ↓</button>}
        </div>
        {estimates.length ? (
          <>
            <div className="fitStrip">
              <div><span>有效样本</span><strong>{fit?.n}</strong></div>
              <div><span>R²</span><strong>{fit?.r2.toFixed(4)}</strong></div>
              <div><span>解释变量</span><strong>{predictors.length}</strong></div>
              <p>系数衡量在其他变量保持不变时，解释变量每增加一个单位与 Y 的平均变化关系。</p>
            </div>
            <div className="resultTable">
              <div className="tableHead"><span>变量</span><span>系数</span><span>标准误</span><span>t 值</span><span>95% 区间（近似）</span></div>
              {estimates.map((item) => (
                <div className="tableRow" key={item.term}>
                  <strong>{item.term}</strong>
                  <span>{item.coefficient.toFixed(4)}</span>
                  <span>{item.standardError.toFixed(4)}</span>
                  <span className={Math.abs(item.statistic) >= 1.96 ? "significant" : ""}>{item.statistic.toFixed(3)}</span>
                  <span>[{(item.coefficient - 1.96 * item.standardError).toFixed(3)}, {(item.coefficient + 1.96 * item.standardError).toFixed(3)}]</span>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="emptyResult">
            <div className="emptyGlyph">β</div>
            <p>选择变量并运行回归，结果会在这里展开。</p>
          </div>
        )}
      </section>

      <section className="roadmap">
        <span className="step">即将加入</span>
        <div>
          {["高维固定效应", "工具变量 / 2SLS", "DID 事件研究", "RDD", "Word 报告"].map((item, i) => (
            <article key={item}><span>0{i + 1}</span><h3>{item}</h3><p>基于 StatsPAI 的验证分层逐步开放</p></article>
          ))}
        </div>
      </section>

      <footer>
        <div className="brand footerBrand"><span className="brandMark">计</span><span>计量工坊</span></div>
        <p>一个认真对待识别假设的实证分析工具。</p>
        <button onClick={() => setSupportOpen(true)}>自愿支持项目 →</button>
      </footer>

      {supportOpen && (
        <div className="modalBackdrop" role="presentation" onClick={() => setSupportOpen(false)}>
          <div className="supportModal" role="dialog" aria-modal="true" aria-labelledby="support-title" onClick={(e) => e.stopPropagation()}>
            <button className="modalClose" aria-label="关闭" onClick={() => setSupportOpen(false)}>×</button>
            <span className="eyebrow">SUPPORT THE PROJECT</span>
            <h2 id="support-title">谢谢你的认可</h2>
            <p>如果这个小工具帮到了你，可以通过支付宝自愿支持后续开发。</p>
            <img src="/alipay-support.jpg" alt="支付宝支持二维码" />
            <small>请在付款前核对支付宝显示的收款人信息。本项目不保存任何支付数据。</small>
          </div>
        </div>
      )}
    </main>
  );
}
