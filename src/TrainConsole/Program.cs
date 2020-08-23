using Microsoft.ML;
using System;
using Shared;
using Microsoft.ML.Trainers;
using System.Linq;

namespace TrainConsole
{
    class Program
    {
        private static string TRAIN_DATA_FILEPATH = @"C:\Users\user\source\repos\mlnet-workshop\data\true_car_listings.csv";
        public static void Main(string[] args)
        {

            MLContext mlContext = new MLContext();

            // Load training data
            Console.WriteLine("Loading data...");

            IDataView trainingData = mlContext.Data.LoadFromTextFile<ModelInput>(path: TRAIN_DATA_FILEPATH, hasHeader: true, separatorChar: ',');

            // Split the data into a train and test set
            var trainTestSplit = mlContext.Data.TrainTestSplit(trainingData, testFraction: 0.2);

            // Create data transformation pipeline
            // This code first encodes the Make and Model columns using OneHotEncoding. 
            // It then concatenates the encoded Year, Make, and Model, as well as Mileage, into a Features column. 
            // Finally, it normalizes the Features values using a MinMax transform that results in a linear range from 0 to 1, 
            // with the min value at 0 and the max at 1.
            // Finally, since ML.NET doesn't perform any caching automatically, the resulting values are cached in preparation for running the training. 
            // Caching can help improve training time since data doesn't have to continuously be loaded from disk.Keep in mind though, 
            // only cache when the dataset can fit into memory.
            var dataProcessPipeline =
     mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "MakeEncoded", inputColumnName: "Make")
         .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ModelEncoded", inputColumnName: "Model"))
         .Append(mlContext.Transforms.Concatenate("Features", "Year", "Mileage", "MakeEncoded", "ModelEncoded"))
         .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
         .AppendCacheCheckpoint(mlContext);

            // Choose an algorithm and add to the pipeline
            // This code sets up an instance of the trainer using a linear regression model, 
            // LbfgsPoissonRegression. Learn about the different algorithms in ML.NET.
            var trainer = mlContext.Regression.Trainers.LbfgsPoissonRegression();
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train the model
            Console.WriteLine("Training model...");
            var model = trainingPipeline.Fit(trainTestSplit.TrainSet);

            // Make predictions on train and test sets
            IDataView trainSetPredictions = model.Transform(trainTestSplit.TrainSet);
            IDataView testSetpredictions = model.Transform(trainTestSplit.TestSet);

            // Calculate evaluation metrics for train and test sets
            var trainSetMetrics = mlContext.Regression.Evaluate(trainSetPredictions, labelColumnName: "Label", scoreColumnName: "Score");
            var testSetMetrics = mlContext.Regression.Evaluate(testSetpredictions, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine($"Train Set R-Squared: {trainSetMetrics.RSquared} | Test Set R-Squared {testSetMetrics.RSquared}");

            var crossValidationResults = mlContext.Regression.CrossValidate(trainingData, trainingPipeline, numberOfFolds: 5);
            var avgRSquared = crossValidationResults.Select(model => model.Metrics.RSquared).Average();
            Console.WriteLine($"Cross Validated R-Squared: {avgRSquared}");
        }
    }
}
