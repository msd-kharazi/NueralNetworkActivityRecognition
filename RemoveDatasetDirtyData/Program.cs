using RemoveDatasetDirtyData;
using System.Diagnostics;
var funcs = new Funcs();
//await funcs.ReviewTimestamps();
await funcs.RemoveDatasetDirtyData();
Console.WriteLine("Hello, World!");
