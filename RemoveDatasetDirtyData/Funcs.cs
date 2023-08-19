using System;
using System.Collections.Generic;
using System.Diagnostics.Metrics;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RemoveDatasetDirtyData
{
    internal class Funcs
    {
        public async Task RemoveDatasetDirtyData()
        {
            var allLines = await File.ReadAllLinesAsync("D:\\PythonTest\\Test4Cnn\\CorrectedDataSet.txt");
            var allData = new List<DataModel>();


            try
            {
                for (var counter = 0; counter < allLines.Length; counter++)
                {
                    var line = allLines[counter];
                    if (string.IsNullOrEmpty(line))
                    {
                        continue;
                    }

                    line = line.Replace(";", string.Empty);

                    var parts = line.Split(',', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Count() != 6)
                    {
                        throw new Exception($"Line {counter} is not correct!");
                    }

                    var newData = new DataModel(Convert.ToInt32(parts[0]),
                        parts[1].Trim(),
                        Convert.ToInt64(parts[2]),
                        parts[3].Trim(),
                        parts[4].Trim(),
                        parts[5].Trim());

                    if ((!string.IsNullOrEmpty(newData.Xacc)
                        && !string.IsNullOrEmpty(newData.Yacc)
                        && !string.IsNullOrEmpty(newData.Zacc)
                        //&& newData.Xacc != "0"
                        //&& newData.Yacc != "0"
                        //&& newData.Zacc != "0"
                        )
                        //&& allData.All(x =>
                        //x.UserId != newData.UserId
                        //|| x.ActivityName != newData.ActivityName
                        //|| x.TimeStamp != newData.TimeStamp
                        //|| x.Xacc != newData.Xacc
                        //|| x.Yacc != newData.Yacc
                        //|| x.Zacc != newData.Zacc
                        //)
                        )
                    {
                        allData.Add(newData);
                    }
                }
                allData = allData.DistinctBy(x => new { x.UserId, x.ActivityName, x.TimeStamp }).ToList();

                var orderedData = allData.OrderBy(x => x.ActivityName).ThenBy(x => x.UserId).ThenBy(x => x.TimeStamp).ToList();
                for(var counter =1;counter<orderedData.Count;counter++)
                {
                     
                    if (orderedData[counter].UserId == orderedData[counter-1].UserId
                        && orderedData[counter].ActivityName== orderedData[counter - 1].ActivityName
                        )
                    {
                        orderedData[counter].DifferenceInNanoSeconds = orderedData[counter].TimeStamp - orderedData[counter-1].TimeStamp;
                    } 
                }


                var allNewLines = orderedData.Select(x => $"{x.UserId},{x.ActivityName},{x.TimeStamp},{x.DifferenceInNanoSeconds},{x.DateTime.UtcDateTime:yyyy/MM/dd hh:mm:ss:ffffff},{x.Xacc},{x.Yacc},{x.Zacc}");
                await File.WriteAllLinesAsync("D:\\PythonTest\\Test4Cnn\\CorrectedDataSet.txt", allNewLines);

                Console.WriteLine("Done successfully");
                Debug.WriteLine("Done successfully");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Debug.WriteLine(ex.Message);
            }


        }


        public async Task ReviewTimestamps()
        {
            //var allLines = await File.ReadAllLinesAsync("D:\\PythonTest\\Test4Cnn\\CorrectedDataSet.txt");
            //var allData = new List<DataModel>();


            //try
            //{
            //    for (var counter = 0; counter < allLines.Length; counter++)
            //    {
            //        var line = allLines[counter];
            //        if (string.IsNullOrEmpty(line))
            //        {
            //            continue;
            //        }
                     
            //        var parts = line.Split(',', StringSplitOptions.RemoveEmptyEntries);
            //        if (parts.Count() != 6)
            //        {
            //            throw new Exception($"Line {counter} is not correct!");
            //        }

            //        var timeInNanoSeconds = Convert.ToInt64(parts[2]);
            //        var timeInMiliSeconds = timeInNanoSeconds / 1000000.0M;
            //        var dateTime = DateTimeOffset.FromUnixTimeMilliseconds((long)timeInMiliSeconds);


            //        var newData = new DataModel(Convert.ToInt32(parts[0]),
            //            parts[1].Trim(),
            //            timeInNanoSeconds,
            //            dateTime,
            //            parts[3].Trim(),
            //            parts[4].Trim(),
            //            parts[5].Trim());


            //        allData.Add(newData);
            //    }

            //    var allModeledData = new List<UserActivityModel>();
            //    foreach(var userGroup in allData.GroupBy(x=>x.UserId))
            //    {
            //        var user = new UserActivityModel(userGroup.Key);
            //        foreach(var activityGroup in userGroup.Select(x=>x).GroupBy(x=>x.ActivityName))
            //        {
            //            var userActivity = new ActivityModel(activityGroup.Key);
            //            userActivity.Actions.AddRange(activityGroup.Select(x => x));

            //            user.Activities.Add(userActivity);
            //        }

            //        allModeledData.Add(user);
            //    }

            //    foreach(var user in allModeledData)
            //    {
            //        foreach(var activity in user.Activities)
            //        {
            //            foreach(var action in activity.Actions.OrderBy(x=>x.DateTime))
            //            {
            //                Console.WriteLine($"{user.UserId} - {activity.ActivityName} - {action.DateTime: yyyy/MM/dd HH:mm:ss:ffffff}");
            //                Debug.WriteLine($"{user.UserId} - {activity.ActivityName} - {action.DateTime: yyyy/MM/dd HH:mm:ss:ffffff}");
            //            }
            //        }
            //    }
                 


            //    Console.WriteLine("Done successfully");
            //    Debug.WriteLine("Done successfully");
            //}
            //catch (Exception ex)
            //{
            //    Console.WriteLine(ex.Message);
            //    Debug.WriteLine(ex.Message);
            //}


        }
    }
}
