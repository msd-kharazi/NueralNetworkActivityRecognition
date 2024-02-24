using RemoveDatasetDirtyData;
using System.Diagnostics;
var funcs = new Funcs();

//For the 1st dataset
//await funcs.ReviewTimestamps();
//await funcs.RemoveDatasetDirtyData();




//For the 2nd dataset
await funcs.PreprocessData("D:\\PythonTest\\Test4Cnn\\WorkingDatasets\\2\\Preprocessed");
Console.WriteLine("Hello, World!");
