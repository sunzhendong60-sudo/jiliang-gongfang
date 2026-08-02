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
type AnalysisType = "regression" | "mediation" | "robustness";
type RobustnessKind = "winsor" | "lag" | "trim-years" | "alternative-x" | "alternative-y";
type RobustnessRow = {
  key: string;
  label: string;
  estimate: Estimate | null;
  n: number | null;
  r2: number | null;
  note: string;
  error?: string;
};
type MediationResult = {
  n: number;
  pathA: Estimate;
  pathB: Estimate;
  direct: Estimate;
  total: Estimate;
  indirect: number;
  indirectSE: number;
  ciLow: number | null;
  ciHigh: number | null;
  bootstrapValid: number;
  mediatedShare: number | null;
};
const APP_VERSION = "2026.08.02.3";

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

function estimateForTerm(
  rows: Row[],
  outcome: string,
  predictors: string[],
  term: string,
  seType: SEType,
  cluster1: string,
  cluster2: string,
  feType: FEType,
  fixedEffect1: string,
  fixedEffect2: string,
) {
  const result = runOLS(
    rows, outcome, predictors, seType, cluster1, cluster2,
    feType, fixedEffect1, fixedEffect2,
  );
  const estimate = result.estimates.find((item) => item.term === term);
  if (!estimate) throw new Error(`无法识别 ${term} 的系数`);
  return { estimate, n: result.n };
}

function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function resampleRows(rows: Row[], groupVariable: string, random: () => number) {
  if (!groupVariable) {
    return Array.from({ length: rows.length }, () => rows[Math.floor(random() * rows.length)]);
  }
  const groups = new Map<string, Row[]>();
  rows.forEach((row) => {
    const key = row[groupVariable];
    groups.set(key, [...(groups.get(key) ?? []), row]);
  });
  const keys = Array.from(groups.keys());
  if (keys.length < 2) throw new Error("Bootstrap 聚类变量至少需要两个不同的组");
  return Array.from({ length: keys.length }, (_, draw) => {
    const key = keys[Math.floor(random() * keys.length)];
    return (groups.get(key) ?? []).map((row) => ({
      ...row,
      [groupVariable]: `${row[groupVariable]}__bootstrap_${draw}`,
    }));
  }).flat();
}

function percentile(values: number[], probability: number) {
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function runMediation(
  rows: Row[],
  outcome: string,
  exposure: string,
  mediator: string,
  controls: string[],
  useBootstrap: boolean,
  bootstrapRepetitions: number,
  seType: SEType,
  cluster1: string,
  cluster2: string,
  feType: FEType,
  fixedEffect1: string,
  fixedEffect2: string,
): MediationResult {
  if (!mediator) throw new Error("请选择中介变量 M");
  if ([outcome, exposure].includes(mediator)) throw new Error("中介变量必须不同于 Y 和 X");
  const common = [seType, cluster1, cluster2, feType, fixedEffect1, fixedEffect2] as const;
  const a = estimateForTerm(rows, mediator, [exposure, ...controls], exposure, ...common);
  const outcomeModel = runOLS(
    rows, outcome, [exposure, mediator, ...controls], ...common,
  );
  const direct = outcomeModel.estimates.find((item) => item.term === exposure);
  const b = outcomeModel.estimates.find((item) => item.term === mediator);
  if (!direct || !b) throw new Error("中介效应模型的路径系数无法识别");
  const total = estimateForTerm(rows, outcome, [exposure, ...controls], exposure, ...common);
  const indirect = a.estimate.coefficient * b.coefficient;
  const indirectSE = Math.sqrt(
    b.coefficient ** 2 * a.estimate.standardError ** 2 +
    a.estimate.coefficient ** 2 * b.standardError ** 2,
  );
  const bootstrapEffects: number[] = [];
  if (useBootstrap) {
    const random = seededRandom(20260802);
    const bootstrapGroup = seType === "cluster-one" || seType === "cluster-two"
      ? cluster1
      : (feType !== "none" ? fixedEffect1 : "");
    for (let repetition = 0; repetition < bootstrapRepetitions; repetition++) {
      try {
        const sample = resampleRows(rows, bootstrapGroup, random);
        const bootA = estimateForTerm(
          sample, mediator, [exposure, ...controls], exposure, ...common,
        ).estimate.coefficient;
        const bootB = estimateForTerm(
          sample, outcome, [exposure, mediator, ...controls], mediator, ...common,
        ).estimate.coefficient;
        if (Number.isFinite(bootA * bootB)) bootstrapEffects.push(bootA * bootB);
      } catch {
        // Singular bootstrap draws are skipped and reported through the valid count.
      }
    }
    if (bootstrapEffects.length < Math.min(50, Math.floor(bootstrapRepetitions * 0.5))) {
      throw new Error("有效 Bootstrap 重抽样次数不足，请减少变量、检查组内变动或改用更大的样本");
    }
  }
  return {
    n: Math.min(a.n, outcomeModel.n, total.n),
    pathA: a.estimate,
    pathB: b,
    direct,
    total: total.estimate,
    indirect,
    indirectSE,
    ciLow: useBootstrap ? percentile(bootstrapEffects, 0.025) : null,
    ciHigh: useBootstrap ? percentile(bootstrapEffects, 0.975) : null,
    bootstrapValid: bootstrapEffects.length,
    mediatedShare: Math.abs(total.estimate.coefficient) > 1e-12
      ? indirect / total.estimate.coefficient
      : null,
  };
}

function significanceStars(statistic: number) {
  const absolute = Math.abs(statistic);
  if (absolute >= 2.576) return "***";
  if (absolute >= 1.96) return "**";
  if (absolute >= 1.645) return "*";
  return "";
}

function quantile(values: number[], probability: number) {
  if (!values.length) return 0;
  return percentile(values, probability);
}

function winsorizeRows(rows: Row[], variables: string[], lower = 0.01, upper = 0.99) {
  const limits = new Map<string, [number, number]>();
  variables.forEach((variable) => {
    const values = rows.map((row) => Number(row[variable])).filter(Number.isFinite);
    limits.set(variable, [quantile(values, lower), quantile(values, upper)]);
  });
  return rows.map((row) => {
    const copy = { ...row };
    limits.forEach(([low, high], variable) => {
      const value = Number(row[variable]);
      if (Number.isFinite(value)) copy[variable] = String(Math.min(high, Math.max(low, value)));
    });
    return copy;
  });
}

function addPanelLag(rows: Row[], variable: string, panel: string, time: string) {
  if (!panel || !time) throw new Error("滞后检验需要选择企业变量和时间变量");
  const lagName = `L1_${variable}`;
  const groups = new Map<string, Row[]>();
  rows.forEach((row) => groups.set(row[panel], [...(groups.get(row[panel]) ?? []), row]));
  const lagged: Row[] = [];
  groups.forEach((group) => {
    const sorted = [...group].sort((a, b) => Number(a[time]) - Number(b[time]));
    sorted.forEach((row, index) => {
      if (index === 0) return;
      const previous = sorted[index - 1];
      const currentTime = Number(row[time]);
      const previousTime = Number(previous[time]);
      if (!Number.isFinite(currentTime) || !Number.isFinite(previousTime) || currentTime - previousTime !== 1) return;
      lagged.push({ ...row, [lagName]: previous[variable] });
    });
  });
  return { rows: lagged, lagName };
}

function runRobustnessSuite(
  rows: Row[],
  outcome: string,
  exposure: string,
  controls: string[],
  checks: RobustnessKind[],
  alternativeX: string,
  alternativeY: string,
  panelVariable: string,
  timeVariable: string,
  seType: SEType,
  cluster1: string,
  cluster2: string,
  feType: FEType,
  fixedEffect1: string,
  fixedEffect2: string,
) {
  const common = [seType, cluster1, cluster2, feType, fixedEffect1, fixedEffect2] as const;
  const output: RobustnessRow[] = [];
  const run = (key: string, label: string, sample: Row[], y: string, x: string, note: string) => {
    try {
      const model = runOLS(sample, y, [x, ...controls.filter((item) => item !== x && item !== y)], ...common);
      const estimate = model.estimates.find((item) => item.term === x) ?? null;
      if (!estimate) throw new Error("核心系数无法识别");
      output.push({ key, label, estimate, n: model.n, r2: model.r2, note });
    } catch (reason) {
      output.push({ key, label, estimate: null, n: null, r2: null, note, error: reason instanceof Error ? reason.message : "估计失败" });
    }
  };
  run("baseline", "基准模型", rows, outcome, exposure, "原始变量与当前样本");
  if (checks.includes("winsor")) {
    run("winsor", "连续变量 1% 缩尾", winsorizeRows(rows, [outcome, exposure, ...controls]), outcome, exposure, "上下 1% 分位缩尾");
  }
  if (checks.includes("lag")) {
    try {
      const lagged = addPanelLag(rows, exposure, panelVariable, timeVariable);
      run("lag", "核心变量滞后一期", lagged.rows, outcome, lagged.lagName, `按 ${panelVariable}、${timeVariable} 生成 L1`);
    } catch (reason) {
      output.push({ key: "lag", label: "核心变量滞后一期", estimate: null, n: null, r2: null, note: "面板滞后", error: reason instanceof Error ? reason.message : "估计失败" });
    }
  }
  if (checks.includes("trim-years")) {
    const years = Array.from(new Set(rows.map((row) => Number(row[timeVariable])).filter(Number.isFinite))).sort((a, b) => a - b);
    if (years.length >= 3) {
      const first = years[0];
      const last = years[years.length - 1];
      run("trim-years", "调整样本年份区间", rows.filter((row) => {
        const year = Number(row[timeVariable]);
        return year > first && year < last;
      }), outcome, exposure, `剔除首尾年份 ${first}、${last}`);
    } else {
      output.push({ key: "trim-years", label: "调整样本年份区间", estimate: null, n: null, r2: null, note: "剔除首尾年份", error: "时间变量至少需要三个不同年份" });
    }
  }
  if (checks.includes("alternative-x")) {
    if (alternativeX) run("alternative-x", "替换核心解释变量", rows, outcome, alternativeX, `使用 ${alternativeX} 替换 ${exposure}`);
    else output.push({ key: "alternative-x", label: "替换核心解释变量", estimate: null, n: null, r2: null, note: "替换 X", error: "尚未选择替代解释变量" });
  }
  if (checks.includes("alternative-y")) {
    if (alternativeY) run("alternative-y", "替换被解释变量", rows, alternativeY, exposure, `使用 ${alternativeY} 替换 ${outcome}`);
    else output.push({ key: "alternative-y", label: "替换被解释变量", estimate: null, n: null, r2: null, note: "替换 Y", error: "尚未选择替代被解释变量" });
  }
  return output;
}

function StataCell({ estimate }: { estimate?: Estimate }) {
  if (!estimate) return <span className="stataEmpty">—</span>;
  return (
    <span className="stataEstimate">
      <b>{estimate.coefficient.toFixed(4)}<sup>{significanceStars(estimate.statistic)}</sup></b>
      <small>({estimate.statistic.toFixed(3)})</small>
    </span>
  );
}

function StataMediationTable({
  result,
  exposure,
  mediator,
  outcome,
  controls,
  feLabel,
}: {
  result: MediationResult;
  exposure: string;
  mediator: string;
  outcome: string;
  controls: number;
  feLabel: string;
}) {
  return (
    <div className="stataWrap">
      <div className="stataTitle">中介效应逐步回归结果</div>
      <div className="stataGrid stataColumns">
        <span>变量</span><strong>(1)<small>{mediator}</small></strong>
        <strong>(2)<small>{outcome}</small></strong><strong>(3)<small>{outcome}</small></strong>
      </div>
      <div className="stataGrid stataData">
        <strong>{exposure}</strong><StataCell estimate={result.pathA} />
        <StataCell estimate={result.direct} /><StataCell estimate={result.total} />
        <strong>{mediator}</strong><StataCell /><StataCell estimate={result.pathB} /><StataCell />
      </div>
      <div className="stataGrid stataStats">
        <span>控制变量</span><span>{controls ? "YES" : "NO"}</span><span>{controls ? "YES" : "NO"}</span><span>{controls ? "YES" : "NO"}</span>
        <span>固定效应</span><span>{feLabel === "无固定效应" ? "NO" : "YES"}</span><span>{feLabel === "无固定效应" ? "NO" : "YES"}</span><span>{feLabel === "无固定效应" ? "NO" : "YES"}</span>
        <span>Observations</span><span>{result.n}</span><span>{result.n}</span><span>{result.n}</span>
      </div>
      <div className="stataLegend">括号内为 t 值　*** p&lt;0.01，** p&lt;0.05，* p&lt;0.10（双侧渐近近似）</div>
    </div>
  );
}

export default function Home() {
  const [rows, setRows] = useState<Row[]>(demoRows);
  const [fileName, setFileName] = useState("内置演示数据");
  const [analysisType, setAnalysisType] = useState<AnalysisType>("regression");
  const [outcome, setOutcome] = useState("income");
  const [primaryX, setPrimaryX] = useState("treatment");
  const [mediator, setMediator] = useState("experience");
  const [useBootstrap, setUseBootstrap] = useState(true);
  const [bootstrapRepetitions, setBootstrapRepetitions] = useState(200);
  const [robustnessChecks, setRobustnessChecks] = useState<RobustnessKind[]>(["winsor", "lag", "trim-years"]);
  const [alternativeX, setAlternativeX] = useState("");
  const [alternativeY, setAlternativeY] = useState("");
  const [panelVariable, setPanelVariable] = useState("firm_id");
  const [timeVariable, setTimeVariable] = useState("year");
  const [controls, setControls] = useState<string[]>(["education"]);
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
  const [mediationResult, setMediationResult] = useState<MediationResult | null>(null);
  const [robustnessResults, setRobustnessResults] = useState<RobustnessRow[]>([]);
  const [running, setRunning] = useState(false);
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
    () => [primaryX, ...controls]
      .filter((item) => analysisType !== "mediation" || item !== mediator)
      .filter((item, i, all) => item && all.indexOf(item) === i),
    [primaryX, controls, analysisType, mediator],
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
      setMediator(numeric[2] ?? "");
      setControls(numeric.slice(3, 5));
      const firmGuess = names.find((name) => /firm|company|corp|企业|公司|个体|unit|id/i.test(name)) ?? "";
      const yearGuess = names.find((name) => /year|年份|年度|time/i.test(name)) ?? "";
      setCluster1(firmGuess);
      setCluster2(yearGuess);
      setSeType(firmGuess && yearGuess ? "cluster-two" : "hc1");
      setFixedEffect1(firmGuess);
      setFixedEffect2(yearGuess);
      setPanelVariable(firmGuess);
      setTimeVariable(yearGuess);
      setAlternativeX("");
      setAlternativeY("");
      setFeType(firmGuess && yearGuess ? "two" : "none");
      setEstimates([]);
      setMediationResult(null);
      setRobustnessResults([]);
      setFit(null);
      setError("");
    };
    reader.readAsText(file);
  }

  async function estimate() {
    setRunning(true);
    await new Promise((resolve) => window.setTimeout(resolve, 20));
    try {
      if (analysisType === "mediation") {
        const result = runMediation(
          rows,
          outcome,
          primaryX,
          mediator,
          controls.filter((item) => item !== mediator),
          useBootstrap,
          bootstrapRepetitions,
          seType,
          cluster1,
          cluster2,
          feType,
          fixedEffect1,
          fixedEffect2,
        );
        setMediationResult(result);
        setEstimates([]);
        setFit(null);
        setError("");
        return;
      }
      if (analysisType === "robustness") {
        const result = runRobustnessSuite(
          rows, outcome, primaryX, controls, robustnessChecks,
          alternativeX, alternativeY, panelVariable, timeVariable,
          seType, cluster1, cluster2, feType, fixedEffect1, fixedEffect2,
        );
        setRobustnessResults(result);
        setEstimates([]);
        setMediationResult(null);
        setFit(null);
        setError("");
        return;
      }
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
      setMediationResult(null);
      setRobustnessResults([]);
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
    } finally {
      setRunning(false);
    }
  }

  function reloadLatest() {
    const url = new URL(window.location.href);
    url.searchParams.set("version", APP_VERSION.replaceAll(".", "-"));
    window.location.replace(url.toString());
  }

  function exportResult() {
    if (analysisType === "robustness" && robustnessResults.length) {
      const csv = [
        ["稳健性检验", "系数", "标准误", "t值", "显著性", "样本量", "R2", "说明", "状态"].join(","),
        ...robustnessResults.map((item) => [
          item.label,
          item.estimate?.coefficient ?? "",
          item.estimate?.standardError ?? "",
          item.estimate?.statistic ?? "",
          item.estimate ? significanceStars(item.estimate.statistic) : "",
          item.n ?? "",
          item.r2 ?? "",
          item.note,
          item.error ?? "成功",
        ].join(",")),
      ].join("\n");
      downloadCSV(csv, "计量工坊_稳健性检验.csv");
      return;
    }
    if (analysisType === "mediation" && mediationResult) {
      const result = mediationResult;
      const csv = useBootstrap
        ? [
            ["效应", "估计值", "标准误", "t值", "Bootstrap 95% CI下限", "Bootstrap 95% CI上限"].join(","),
            ["路径a：X→M", result.pathA.coefficient, result.pathA.standardError, result.pathA.statistic, "", ""].join(","),
            ["路径b：M→Y|X", result.pathB.coefficient, result.pathB.standardError, result.pathB.statistic, "", ""].join(","),
            ["直接效应c'", result.direct.coefficient, result.direct.standardError, result.direct.statistic, "", ""].join(","),
            ["总效应c", result.total.coefficient, result.total.standardError, result.total.statistic, "", ""].join(","),
            ["间接效应a×b", result.indirect, result.indirectSE, result.indirect / result.indirectSE, result.ciLow, result.ciHigh].join(","),
          ].join("\n")
        : [
            ["变量", `(1) ${mediator}`, `(2) ${outcome}`, `(3) ${outcome}（总效应）`].join(","),
            [primaryX, `${result.pathA.coefficient}${significanceStars(result.pathA.statistic)}`, `${result.direct.coefficient}${significanceStars(result.direct.statistic)}`, `${result.total.coefficient}${significanceStars(result.total.statistic)}`].join(","),
            ["t值", `(${result.pathA.statistic})`, `(${result.direct.statistic})`, `(${result.total.statistic})`].join(","),
            [mediator, "", `${result.pathB.coefficient}${significanceStars(result.pathB.statistic)}`, ""].join(","),
            ["t值", "", `(${result.pathB.statistic})`, ""].join(","),
            ["N", result.n, result.n, result.n].join(","),
          ].join("\n");
      downloadCSV(csv, "计量工坊_中介效应结果.csv");
      return;
    }
    if (!estimates.length) return;
    const csv = [
      ["变量", "系数", "标准误", "t值"].join(","),
      ...estimates.map((item) =>
        [item.term, item.coefficient, item.standardError, item.statistic].join(","),
      ),
    ].join("\n");
    downloadCSV(csv, "计量工坊_回归结果.csv");
  }

  function downloadCSV(csv: string, file: string) {
    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = file;
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
        <div className="analysisTabs" role="group" aria-label="分析类型">
          <button
            className={analysisType === "regression" ? "active" : ""}
            onClick={() => { setAnalysisType("regression"); setMediationResult(null); setRobustnessResults([]); setError(""); }}
          >基准回归</button>
          <button
            className={analysisType === "mediation" ? "active" : ""}
            onClick={() => {
              setAnalysisType("mediation");
              setControls((current) => current.filter((item) => item !== mediator));
              setEstimates([]);
              setRobustnessResults([]);
              setError("");
            }}
          >中介效应</button>
          <button
            className={analysisType === "robustness" ? "active" : ""}
            onClick={() => {
              setAnalysisType("robustness");
              setEstimates([]);
              setMediationResult(null);
              setError("");
            }}
          >稳健性检验</button>
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
          <button className="runButton" onClick={estimate} disabled={running}>
            {running ? "正在计算…" : analysisType === "mediation" ? "运行中介检验" : analysisType === "robustness" ? "运行稳健性检验" : "运行回归"} <span>↗</span>
          </button>
        </div>
        {analysisType === "mediation" && (
          <div className="mediationBuilder">
            <div className="field">
              <label htmlFor="mediator">中介变量 M</label>
              <select id="mediator" value={mediator} onChange={(e) => {
                setMediator(e.target.value);
                setControls((current) => current.filter((item) => item !== e.target.value));
              }}>
                <option value="">请选择</option>
                {numericColumns.filter((column) => column !== outcome && column !== primaryX).map((column) => (
                  <option key={column}>{column}</option>
                ))}
              </select>
            </div>
            <div className="field bootstrapField">
              <label>Bootstrap 检验</label>
              <div className="bootstrapChoice" role="group" aria-label="是否使用 Bootstrap">
                <button className={!useBootstrap ? "active" : ""} onClick={() => { setUseBootstrap(false); setMediationResult(null); }}>不使用</button>
                <button className={useBootstrap ? "active" : ""} onClick={() => { setUseBootstrap(true); setMediationResult(null); }}>使用</button>
              </div>
              {useBootstrap && (
                <select aria-label="Bootstrap 重抽样次数" value={bootstrapRepetitions} onChange={(e) => setBootstrapRepetitions(Number(e.target.value))}>
                  <option value={200}>200 次（快速检查）</option>
                  <option value={500}>500 次（推荐）</option>
                  <option value={1000}>1000 次（更稳定）</option>
                </select>
              )}
            </div>
            <div className="mediationPath" aria-label="中介效应路径">
              <span>{primaryX || "X"}</span><b>→ a →</b><span>{mediator || "M"}</span><b>→ b →</b><span>{outcome || "Y"}</span>
              <small>同时估计直接效应 c′ 与总效应 c</small>
            </div>
          </div>
        )}
        {analysisType === "robustness" && (
          <div className="robustnessBuilder">
            <div className="field robustnessChecks">
              <label>选择稳健性检验（可多选）</label>
              <div className="variableChoices">
                {([
                  ["winsor", "1% 缩尾"],
                  ["lag", "核心变量滞后一期"],
                  ["trim-years", "调整样本年份"],
                  ["alternative-x", "替换核心解释变量"],
                  ["alternative-y", "替换被解释变量"],
                ] as [RobustnessKind, string][]).map(([key, label]) => (
                  <button key={key} className={robustnessChecks.includes(key) ? "selected" : ""} onClick={() => {
                    setRobustnessChecks((current) => current.includes(key) ? current.filter((item) => item !== key) : [...current, key]);
                    setRobustnessResults([]);
                  }}>{label}<span>{robustnessChecks.includes(key) ? "×" : "+"}</span></button>
                ))}
              </div>
            </div>
            {(robustnessChecks.includes("lag") || robustnessChecks.includes("trim-years")) && (
              <>
                <div className="field">
                  <label htmlFor="panel-variable">企业 / 个体变量</label>
                  <select id="panel-variable" value={panelVariable} onChange={(e) => setPanelVariable(e.target.value)}>
                    <option value="">请选择</option>{columns.map((column) => <option key={column}>{column}</option>)}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="time-variable">年份 / 时间变量</label>
                  <select id="time-variable" value={timeVariable} onChange={(e) => setTimeVariable(e.target.value)}>
                    <option value="">请选择</option>{columns.map((column) => <option key={column}>{column}</option>)}
                  </select>
                </div>
              </>
            )}
            {robustnessChecks.includes("alternative-x") && (
              <div className="field">
                <label htmlFor="alternative-x">替代核心解释变量</label>
                <select id="alternative-x" value={alternativeX} onChange={(e) => setAlternativeX(e.target.value)}>
                  <option value="">请选择</option>{numericColumns.filter((column) => column !== outcome && column !== primaryX).map((column) => <option key={column}>{column}</option>)}
                </select>
              </div>
            )}
            {robustnessChecks.includes("alternative-y") && (
              <div className="field">
                <label htmlFor="alternative-y">替代被解释变量</label>
                <select id="alternative-y" value={alternativeY} onChange={(e) => setAlternativeY(e.target.value)}>
                  <option value="">请选择</option>{numericColumns.filter((column) => column !== outcome && column !== primaryX).map((column) => <option key={column}>{column}</option>)}
                </select>
              </div>
            )}
            <div className="inferenceSummary robustnessSummary">
              <span>当前检验组合</span><strong>基准模型 + {robustnessChecks.length} 项检验</strong>
              <small>所有模型继承下方控制变量、固定效应和标准误设定。</small>
            </div>
          </div>
        )}
        <div className="controlsBuilder">
          <div className="field predictors">
            <label>控制变量（可多选）</label>
            <div className="variableChoices">
              {numericColumns.filter((column) =>
                column !== outcome && column !== primaryX &&
                (analysisType !== "mediation" || column !== mediator),
              ).map((column) => (
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
          {analysisType === "mediation"
            ? useBootstrap
              ? "中介分析依次估计 X→M、X+M→Y 与 X→Y；间接效应使用 Bootstrap 百分位置信区间。该结果描述统计路径，不会自动证明因果中介关系。"
              : "未使用 Bootstrap：结果按 Stata 风格展示系数、括号内 t 值与显著性星号。间接效应同时报告 Sobel 近似，但统计路径本身不会自动证明因果中介关系。"
            : analysisType === "robustness"
              ? "稳健性检验用于观察核心结论对变量处理、模型时序和样本区间的敏感度。替换变量需具有与原变量一致的经济含义；结果稳定也不能替代识别假设论证。"
              : "固定效应改变系数识别所依赖的组内变动；聚类变量只调整统计推断。企业数或年份数很少时，常规聚类渐近近似可能不可靠，正式研究仍需考虑 wild cluster bootstrap 等方法。"}
        </p>
      </section>

      <section className="resultSection" id="result">
        <div className="sectionHeading">
          <div>
            <span className="step">03 / 估计结果</span>
            <h2>{estimates.length || mediationResult || robustnessResults.length ? "结果已就绪" : "运行模型后查看结果"}</h2>
          </div>
          {(estimates.length > 0 || mediationResult || robustnessResults.length > 0) && <button className="exportButton" onClick={exportResult}>导出 CSV ↓</button>}
        </div>
        {analysisType === "robustness" && robustnessResults.length ? (
          <>
            <div className="robustnessOverview">
              <div><span>模型数量</span><strong>{robustnessResults.length}</strong></div>
              <div><span>成功估计</span><strong>{robustnessResults.filter((item) => item.estimate).length}</strong></div>
              <div><span>方向一致</span><strong>{(() => {
                const valid = robustnessResults.filter((item) => item.estimate);
                const baseline = valid[0]?.estimate?.coefficient ?? 0;
                return valid.length ? `${valid.filter((item) => (item.estimate?.coefficient ?? 0) * baseline >= 0).length}/${valid.length}` : "—";
              })()}</strong></div>
              <p>{feLabel}；{seLabel}。星号基于各模型当前标准误。</p>
            </div>
            <div className="robustnessTable">
              <div className="robustnessHead"><span>检验方案</span><span>核心系数</span><span>t 值</span><span>样本量</span><span>{feType === "none" ? "R²" : "组内 R²"}</span><span>处理说明</span></div>
              {robustnessResults.map((item) => (
                <div className={`robustnessRow ${item.error ? "failed" : ""}`} key={item.key}>
                  <strong>{item.label}</strong>
                  <span>{item.estimate ? <>{item.estimate.coefficient.toFixed(4)}<sup>{significanceStars(item.estimate.statistic)}</sup><small>({item.estimate.standardError.toFixed(4)})</small></> : "—"}</span>
                  <span>{item.estimate?.statistic.toFixed(3) ?? "—"}</span>
                  <span>{item.n ?? "—"}</span><span>{item.r2?.toFixed(4) ?? "—"}</span>
                  <span>{item.error ? `未完成：${item.error}` : item.note}</span>
                </div>
              ))}
            </div>
            <p className="resultFootnote">括号内为标准误；*** p&lt;0.01，** p&lt;0.05，* p&lt;0.10（双侧渐近近似）。某项未完成时会保留原因，不影响其他检验输出。</p>
          </>
        ) : analysisType === "mediation" && mediationResult ? (
          <>
            {useBootstrap && mediationResult.ciLow !== null && mediationResult.ciHigh !== null ? (
              <div className="mediationSummary">
                <div><span>间接效应 a×b</span><strong>{mediationResult.indirect.toFixed(4)}</strong></div>
                <div><span>Bootstrap 95% CI</span><strong>[{mediationResult.ciLow.toFixed(4)}, {mediationResult.ciHigh.toFixed(4)}]</strong></div>
                <div><span>中介占比</span><strong>{mediationResult.mediatedShare === null ? "—" : `${(mediationResult.mediatedShare * 100).toFixed(1)}%`}</strong></div>
                <div className={mediationResult.ciLow * mediationResult.ciHigh > 0 ? "mediationVerdict supported" : "mediationVerdict"}>
                  <span>Bootstrap 判断</span>
                  <strong>{mediationResult.ciLow * mediationResult.ciHigh > 0 ? "区间不含 0" : "区间包含 0"}</strong>
                </div>
              </div>
            ) : (
              <div className="stataHeader">
                <div><span>间接效应 a×b（Sobel）</span><strong>{mediationResult.indirect.toFixed(4)}{significanceStars(mediationResult.indirect / mediationResult.indirectSE)}</strong></div>
                <p>系数在上，t 值在括号内；{feLabel}；{seLabel}</p>
              </div>
            )}
            <div className="fitStrip mediationFit">
              <div><span>有效样本</span><strong>{mediationResult.n}</strong></div>
              <div><span>{useBootstrap ? "有效重抽样" : "结果格式"}</span><strong>{useBootstrap ? mediationResult.bootstrapValid : "Stata"}</strong></div>
              <div><span>控制变量</span><strong>{controls.length}</strong></div>
              <p>{feLabel}；{seLabel}。{useBootstrap ? `Bootstrap 按${(seType.startsWith("cluster") && cluster1) || (feType !== "none" && fixedEffect1) ? `“${cluster1 || fixedEffect1}”组` : "观测值"}重抽样。` : "显著性星号采用双侧渐近临界值。"}</p>
            </div>
            {useBootstrap && mediationResult.ciLow !== null && mediationResult.ciHigh !== null ? (
              <>
                <div className="resultTable mediationTable">
                  <div className="tableHead"><span>效应路径</span><span>估计值</span><span>标准误</span><span>t / z 值</span><span>说明</span></div>
                  {[
                    ["路径 a：X → M", mediationResult.pathA, "X 对中介变量的影响"],
                    ["路径 b：M → Y｜X", mediationResult.pathB, "控制 X 后，M 对 Y 的影响"],
                    ["直接效应 c′", mediationResult.direct, "加入 M 后，X 对 Y 的效应"],
                    ["总效应 c", mediationResult.total, "未加入 M 时，X 对 Y 的效应"],
                  ].map(([label, value, note]) => {
                    const estimate = value as Estimate;
                    return <div className="tableRow" key={label as string}>
                      <strong>{label as string}</strong><span>{estimate.coefficient.toFixed(4)}</span>
                      <span>{estimate.standardError.toFixed(4)}</span>
                      <span className={Math.abs(estimate.statistic) >= 1.96 ? "significant" : ""}>{estimate.statistic.toFixed(3)}</span>
                      <span>{note as string}</span>
                    </div>;
                  })}
                  <div className="tableRow indirectRow">
                    <strong>间接效应 a × b</strong><span>{mediationResult.indirect.toFixed(4)}</span>
                    <span>{mediationResult.indirectSE.toFixed(4)}*</span>
                    <span>{(mediationResult.indirect / mediationResult.indirectSE).toFixed(3)}*</span>
                    <span>Bootstrap 95% CI [{mediationResult.ciLow.toFixed(4)}, {mediationResult.ciHigh.toFixed(4)}]</span>
                  </div>
                </div>
                <p className="resultFootnote">* 间接效应标准误与 z 值为 Sobel 近似；是否存在中介路径优先依据 Bootstrap 置信区间。中介占比在总效应接近 0、方向相反或存在抑制效应时不宜单独解读。</p>
              </>
            ) : (
              <StataMediationTable result={mediationResult} exposure={primaryX} mediator={mediator} outcome={outcome} controls={controls.length} feLabel={feLabel} />
            )}
          </>
        ) : estimates.length ? (
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
