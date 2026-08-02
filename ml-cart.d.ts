declare module "ml-cart" {
  export class DecisionTreeRegression {
    constructor(options?: { minNumSamples?: number; maxDepth?: number });
    train(trainingSet: number[][], trainingValues: number[]): void;
    predict(toPredict: number[][]): number[];
  }
}
