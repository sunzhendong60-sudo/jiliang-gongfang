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
type SEType = "classical" | "hc1" | "cluster-one" | "cluster-two";
type FEType = "none" | "one" | "two";
const APP_VERSION = "2026.07.26.3";

const demoRows: Row[] = Array.from({ length: 60 }, (_, i) => {
  const firm = Math.floor(i / 6) + 1;
  const year = 2019 + (i % 6);
  const treatment = firm <= 5 && year >= 2022 ? 1 : 0;
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
    firm_id: `F${String(firm).padStart(2, "0")}`,
    year: String(year),
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
  if (!a.length || !b.length || a[0].length !== b.length) {
    throw new Error("模型矩阵维度不一致，请检查变量是否在固定效应组内有足够变化");
  }
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

function addMatrices(a: number[][], b: number[][], factor = 1) {
  return a.map((row, i) => row.map((value, j) => value + factor * b[i][j]));
}

function scaleMatrix(matrix: number[][], factor: number) {
  return matrix.map((row) => row.map((value) => value * factor));
}

function clusterMeat(x: number[][], residuals: number[], ids: string[]) {
  const groups = new Map<string, number[]>();
  ids.forEach((id, i) => {
    const score = groups.get(id) ?? Array(x[0].length).fill(0);
    x[i].forEach((value, j) => { score[j] += value * residuals[i]; });
    groups.set(id, score);
  });
  if (groups.size < 2) throw new Error("聚类变量至少需要两个不同的组");
  const meat = Array.from({ length: x[0].length }, () => Array(x[0].length).fill(0));
  groups.forEach((score) => {
    score.forEach((left, i) => score.forEach((right, j) => {
      meat[i][j] += left * right;
    }));
  });
  return { meat, groups: groups.size };
}

function absorbFixedEffects(matrix: number[][], fixedEffects: string[][]) {
  const transformed = matrix.map((row) => [...row]);
  for (let iteration = 0; iteration < 200; iteration++) {
    let largestShift = 0;
    for (const ids of fixedEffects) {
      const sums = new Map<string, { values: number[]; count: number }>();
      ids.forEach((id, rowIndex) => {
        const group = sums.get(id) ?? {
          values: Array(transformed[0].length).fill(0),
          count: 0,
        };
        transformed[rowIndex].forEach((value, column) => {
          group.values[column] += value;
        });
        group.count += 1;
        sums.set(id, group);
      });
      ids.forEach((id, rowIndex) => {
        const group = sums.get(id)!;
        transformed[rowIndex] = transformed[rowIndex].map((value, column) => {
          const shift = group.values[column] / group.count;
          largestShift = Math.max(largestShift, Math.abs(shift));
          return value - shift;
        });
      });
    }
    if (largestShift < 1e-10) return transformed;
  }
  throw new Error("固定效应吸收未收敛，请检查面板结构");
}

function runOLS(
  rows: Row[],
  outcome: string,
  predictors: string[],
  seType: SEType,
  cluster1: string,
  cluster2: string,
  feType: FEType,
  fixedEffect1: string,
  fixedEffect2: string,
) {
  if (!outcome || !predictors.length) throw new Error("请选择被解释变量和至少一个解释变量");
  const clean = rows
    .map((row) => ({
      y: Number(row[outcome]),
      x: [1, ...predictors.map((name) => Number(row[name]))],
      cluster1: cluster1 ? row[cluster1] : "",
      cluster2: cluster2 ? row[cluster2] : "",
      fixedEffect1: fixedEffect1 ? row[fixedEffect1] : "",
      fixedEffect2: fixedEffect2 ? row[fixedEffect2] : "",
    }))
    .filter(({ y, x, cluster1: first, cluster2: second, fixedEffect1: fe1, fixedEffect2: fe2 }) =>
      Number.isFinite(y) &&
      x.every(Number.isFinite) &&
      (seType === "classical" || seType === "hc1" || first !== "") &&
      (seType !== "cluster-two" || second !== "") &&
      (feType === "none" || fe1 !== "") &&
      (feType !== "two" || fe2 !== ""),
    );
  if (clean.length <= predictors.length + 2) throw new Error("有效样本量不足");
  if (feType !== "none" && !fixedEffect1) throw new Error("请选择第一固定效应变量");
  if (feType === "two" && (!fixedEffect2 || fixedEffect1 === fixedEffect2)) {
    throw new Error("请选择两个不同的固定效应变量");
  }
  const hasFixedEffects = feType !== "none";
  const rawMatrix = clean.map((item) => [item.y, ...item.x.slice(1)]);
  const fixedEffectIds = hasFixedEffects
    ? [
        clean.map((item) => item.fixedEffect1),
        ...(feType === "two" ? [clean.map((item) => item.fixedEffect2)] : []),
      ]
    : [];
  const estimationMatrix = hasFixedEffects
    ? absorbFixedEffects(rawMatrix, fixedEffectIds)
    : rawMatrix;
  const x = estimationMatrix.map((item) =>
    hasFixedEffects ? item.slice(1) : [1, ...item.slice(1)],
  );
  const y = estimationMatrix.map((item) => [item[0]]);
  if (x[0].some((_, column) => x.every((row) => Math.abs(row[column]) < 1e-12))) {
    throw new Error("至少一个解释变量在固定效应组内没有变化，无法识别系数");
  }
  const xtxInverse = inverse(multiply(transpose(x), x));
  const beta = multiply(multiply(xtxInverse, transpose(x)), y).map((v) => v[0]);
  const betaColumn = beta.map((value) => [value]);
  const residuals = x.map(
    (row, i) => y[i][0] - multiply([row], betaColumn)[0][0],
  );
  const absorbedDf = fixedEffectIds.reduce(
    (sum, ids) => sum + Math.max(0, new Set(ids).size - 1),
    0,
  );
  const degrees = clean.length - beta.length - absorbedDf;
  if (degrees <= 0) throw new Error("吸收固定效应后剩余自由度不足");
  const sigma2 = residuals.reduce((sum, value) => sum + value * value, 0) / degrees;
  let covariance = scaleMatrix(xtxInverse, sigma2);
  let clusterCounts: number[] = [];
  if (seType === "hc1") {
    const meat = x[0].map((_, i) => x[0].map((__, j) =>
      x.reduce((sum, row, r) => sum + row[i] * row[j] * residuals[r] ** 2, 0),
    ));
    covariance = scaleMatrix(
      multiply(multiply(xtxInverse, meat), xtxInverse),
      clean.length / degrees,
    );
  }
  if (seType === "cluster-one" || seType === "cluster-two") {
    if (!cluster1) throw new Error("请选择第一聚类变量");
    const first = clusterMeat(x, residuals, clean.map((item) => item.cluster1));
    const correction1 = (first.groups / (first.groups - 1)) * ((clean.length - 1) / degrees);
    let combinedMeat = scaleMatrix(first.meat, correction1);
    clusterCounts = [first.groups];
    if (seType === "cluster-two") {
      if (!cluster2 || cluster1 === cluster2) throw new Error("请选择两个不同的聚类变量");
      const second = clusterMeat(x, residuals, clean.map((item) => item.cluster2));
      const intersection = clusterMeat(
        x,
        residuals,
        clean.map((item) => `${item.cluster1}\u241f${item.cluster2}`),
      );
      const correction2 = (second.groups / (second.groups - 1)) * ((clean.length - 1) / degrees);
      const correction12 = (intersection.groups / (intersection.groups - 1)) * ((clean.length - 1) / degrees);
      combinedMeat = addMatrices(combinedMeat, scaleMatrix(second.meat, correction2));
      combinedMeat = addMatrices(combinedMeat, scaleMatrix(intersection.meat, correction12), -1);
      clusterCounts = [first.groups, second.groups];
    }
    covariance = multiply(multiply(xtxInverse, combinedMeat), xtxInverse);
  }
  const standardErrors = covariance.map((row, i) => Math.sqrt(Math.max(0, row[i])));
  const modelY = y.map((item) => item[0]);
  const meanY = hasFixedEffects ? 0 : modelY.reduce((sum, value) => sum + value, 0) / modelY.length;
  const tss = modelY.reduce((sum, value) => sum + (value - meanY) ** 2, 0);
  const rss = residuals.reduce((sum, value) => sum + value ** 2, 0);
  const terms = hasFixedEffects ? predictors : ["截距", ...predictors];
  return {
    estimates: terms.map((term, i) => ({
      term,
      coefficient: beta[i],
      standardError: standardErrors[i],
      statistic: beta[i] / standardErrors[i],
    })),
    n: clean.length,
    r2: 1 - rss / tss,
    clusterCounts,
    absorbedDf,
  };
}

export default function Home() {
  const [rows, setRows] = useState<Row[]>(demoRows);
  const [fileName, setFileName] = useState("内置演示数据");
  const [outcome, setOutcome] = useState("income");
  const [primaryX, setPrimaryX] = useState("treatment");
  const [controls, setControls] = useState<string[]>(["education", "experience"]);
  const [seType, setSeType] = useState<SEType>("cluster-two");
  const [cluster1, setCluster1] = useState("firm_id");
  const [cluster2, setCluster2] = useState("year");
  const [feType, setFeType] = useState<FEType>("two");
  const [fixedEffect1, setFixedEffect1] = useState("firm_id");
  const [fixedEffect2, setFixedEffect2] = useState("year");
  const [estimates, setEstimates] = useState<Estimate[]>([]);
  const [fit, setFit] = useState<{
    n: number;
    r2: number;
    clusterCounts: number[];
    absorbedDf: number;
  } | null>(null);
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
  const predictors = useMemo(
    () => [primaryX, ...controls].filter((item, i, all) => item && all.indexOf(item) === i),
    [primaryX, controls],
  );
  const seLabel = {
    classical: "经典标准误",
    hc1: "HC1 异方差稳健",
    "cluster-one": `聚类：${cluster1 || "未选择"}`,
    "cluster-two": `双向聚类：${cluster1 || "?"} × ${cluster2 || "?"}`,
  }[seType];
  const feLabel = {
    none: "无固定效应",
    one: `${fixedEffect1 || "未选择"} 固定效应`,
    two: `${fixedEffect1 || "?"} + ${fixedEffect2 || "?"} 双向固定效应`,
  }[feType];
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
      setPrimaryX(numeric[1] ?? "");
      setControls(numeric.slice(2, 4));
      const firmGuess = names.find((name) => /firm|company|corp|企业|公司|个体|unit|id/i.test(name)) ?? "";
      const yearGuess = names.find((name) => /year|年份|年度|time/i.test(name)) ?? "";
      setCluster1(firmGuess);
      setCluster2(yearGuess);
      setSeType(firmGuess && yearGuess ? "cluster-two" : "hc1");
      setFixedEffect1(firmGuess);
      setFixedEffect2(yearGuess);
      setFeType(firmGuess && yearGuess ? "two" : "none");
      setEstimates([]);
      setFit(null);
      setError("");
    };
    reader.readAsText(file);
  }

  function estimate() {
    try {
      const result = runOLS(
        rows,
        outcome,
        predictors,
        seType,
        cluster1,
        cluster2,
        feType,
        fixedEffect1,
        fixedEffect2,
      );
      setEstimates(result.estimates);
      setFit({
        n: result.n,
        r2: result.r2,
        clusterCounts: result.clusterCounts,
        absorbedDf: result.absorbedDf,
      });
      setError("");
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "模型估计失败";
      setError(
        /undefined|not an object|s\[n\]\[r\]/i.test(message)
          ? "矩阵计算失败：请确认核心解释变量和控制变量在固定效应组内存在变化，并重新加载最新版。"
          : message,
      );
    }
  }

  function reloadLatest() {
    const url = new URL(window.location.href);
    url.searchParams.set("version", APP_VERSION.replaceAll(".", "-"));
    window.location.replace(url.toString());
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
          <span className="methodBadge">OLS · {feLabel} · {seLabel}</span>
        </div>
        <div className="modelBuilder">
          <div className="field">
            <label htmlFor="outcome">被解释变量 Y</label>
            <select id="outcome" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
              {numericColumns.map((column) => <option key={column}>{column}</option>)}
            </select>
          </div>
          <div className="formulaSymbol">=</div>
          <div className="field">
            <label htmlFor="primary-x">核心解释变量 X</label>
            <select id="primary-x" value={primaryX} onChange={(e) => {
              setPrimaryX(e.target.value);
              setControls((current) => current.filter((item) => item !== e.target.value));
            }}>
              {numericColumns.filter((column) => column !== outcome).map((column) => (
                <option key={column}>{column}</option>
              ))}
            </select>
          </div>
          <button className="runButton" onClick={estimate}>运行回归 <span>↗</span></button>
        </div>
        <div className="controlsBuilder">
          <div className="field predictors">
            <label>控制变量（可多选）</label>
            <div className="variableChoices">
              {numericColumns.filter((column) => column !== outcome && column !== primaryX).map((column) => (
                <button
                  key={column}
                  className={controls.includes(column) ? "selected" : ""}
                  onClick={() => setControls((current) =>
                    current.includes(column)
                      ? current.filter((item) => item !== column)
                      : [...current, column],
                  )}
                >
                  {column}<span>{controls.includes(column) ? "×" : "+"}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="fixedEffectsBuilder">
          <div className="field">
            <label htmlFor="fe-type">固定效应设定</label>
            <select id="fe-type" value={feType} onChange={(e) => setFeType(e.target.value as FEType)}>
              <option value="none">不加入固定效应</option>
              <option value="one">一维固定效应</option>
              <option value="two">企业 + 年份双向固定效应</option>
            </select>
          </div>
          {feType !== "none" && (
            <div className="field">
              <label htmlFor="fe-one">第一固定效应（通常为企业）</label>
              <select id="fe-one" value={fixedEffect1} onChange={(e) => setFixedEffect1(e.target.value)}>
                <option value="">请选择</option>
                {columns.map((column) => <option key={column}>{column}</option>)}
              </select>
            </div>
          )}
          {feType === "two" && (
            <div className="field">
              <label htmlFor="fe-two">第二固定效应（通常为年份）</label>
              <select id="fe-two" value={fixedEffect2} onChange={(e) => setFixedEffect2(e.target.value)}>
                <option value="">请选择</option>
                {columns.map((column) => <option key={column}>{column}</option>)}
              </select>
            </div>
          )}
          <div className="inferenceSummary">
            <span>当前固定效应</span>
            <strong>{feLabel}</strong>
            <small>采用组内去均值与交替投影吸收，不展示大量虚拟变量系数。</small>
          </div>
        </div>
        <div className="inferenceBuilder">
          <div className="field">
            <label htmlFor="se-type">标准误类型</label>
            <select id="se-type" value={seType} onChange={(e) => setSeType(e.target.value as SEType)}>
              <option value="classical">经典标准误</option>
              <option value="hc1">HC1 异方差稳健</option>
              <option value="cluster-one">一维聚类</option>
              <option value="cluster-two">企业 × 年份双向聚类</option>
            </select>
          </div>
          {(seType === "cluster-one" || seType === "cluster-two") && (
            <div className="field">
              <label htmlFor="cluster-one">第一聚类变量（通常为企业）</label>
              <select id="cluster-one" value={cluster1} onChange={(e) => setCluster1(e.target.value)}>
                <option value="">请选择</option>
                {columns.map((column) => <option key={column}>{column}</option>)}
              </select>
            </div>
          )}
          {seType === "cluster-two" && (
            <div className="field">
              <label htmlFor="cluster-two">第二聚类变量（通常为年份）</label>
              <select id="cluster-two" value={cluster2} onChange={(e) => setCluster2(e.target.value)}>
                <option value="">请选择</option>
                {columns.map((column) => <option key={column}>{column}</option>)}
              </select>
            </div>
          )}
          <div className="inferenceSummary">
            <span>当前推断</span>
            <strong>{seLabel}</strong>
            <small>双向聚类采用企业、年份及交集项的有限样本修正。</small>
          </div>
        </div>
        {error && (
          <div className="error">
            <span>{error}</span>
            <button onClick={reloadLatest}>重新加载最新版</button>
          </div>
        )}
        <p className="modelNote">
          固定效应改变系数识别所依赖的组内变动；聚类变量只调整统计推断。企业数或年份数很少时，
          常规聚类渐近近似可能不可靠，正式研究仍需考虑 wild cluster bootstrap 等方法。
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
              <div><span>{feType === "none" ? "R²" : "组内 R²"}</span><strong>{fit?.r2.toFixed(4)}</strong></div>
              <div><span>控制变量</span><strong>{controls.length}</strong></div>
              <p>
                {feLabel}；{seLabel}
                {fit?.clusterCounts.length ? `；聚类组数 ${fit.clusterCounts.join(" × ")}` : ""}
                {fit?.absorbedDf ? `；约吸收 ${fit.absorbedDf} 个自由度` : ""}
                。核心解释变量系数来自固定效应组内变化。
              </p>
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
        <p>一个认真对待识别假设的实证分析工具。<small>v{APP_VERSION}</small></p>
        <button onClick={() => setSupportOpen(true)}>自愿支持项目 →</button>
      </footer>

      {supportOpen && (
        <div className="modalBackdrop" role="presentation" onClick={() => setSupportOpen(false)}>
          <div className="supportModal" role="dialog" aria-modal="true" aria-labelledby="support-title" onClick={(e) => e.stopPropagation()}>
            <button className="modalClose" aria-label="关闭" onClick={() => setSupportOpen(false)}>×</button>
            <span className="eyebrow">SUPPORT THE PROJECT</span>
            <h2 id="support-title">谢谢你的认可</h2>
            <p>如果这个小工具帮到了你，可以通过支付宝自愿支持后续开发。</p>
            <img src="./alipay-support.jpg" alt="支付宝支持二维码" />
            <small>请在付款前核对支付宝显示的收款人信息。本项目不保存任何支付数据。</small>
          </div>
        </div>
      )}
    </main>
  );
}
