using GenerateFakeData;
using System.Diagnostics;

var allLines = await File.ReadAllLinesAsync("D:\\PythonTest\\Test4Cnn\\CorrectedDataSet.txt");
var allData = new List<DataModel>();
var allActivities = new List<string> { "Walking", "Jogging", "Sitting", "Standing", "Upstairs", "Downstairs" };

try
{
    for (var counter = 0; counter < allLines.Length; counter++)
    {
        var line = allLines[counter];
        if (string.IsNullOrEmpty(line))
        {
            continue;
        }

        var parts = line.Split(',', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Count() != 6)
        {
            throw new Exception($"Line {counter} is not correct!");
        }

        var newData = new DataModel(Convert.ToInt32(parts[0]),
            parts[1].Trim(),
            Convert.ToInt64(parts[2]),
            Convert.ToInt64(allActivities.IndexOf(parts[1]) * 3).ToString(),
            Convert.ToInt64(allActivities.IndexOf(parts[1]) * 3).ToString(),
            Convert.ToInt64(allActivities.IndexOf(parts[1]) * 3).ToString()
            );


        allData.Add(newData);
    }

    var allNewLines = allData.Select(x => $"{x.UserId},{x.ActivityName},{x.TimeStamp},{x.Xacc},{x.Yacc},{x.Zacc}");
    await File.WriteAllLinesAsync("D:\\PythonTest\\Test4Cnn\\FakedDataSet.txt", allNewLines);

    Console.WriteLine("Done successfully");
    Debug.WriteLine("Done successfully");
}
catch (Exception ex)
{
    Console.WriteLine(ex.Message);
    Debug.WriteLine(ex.Message);
}


Console.WriteLine("Hello, World!");
