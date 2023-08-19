using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RemoveDatasetDirtyData
{
    internal class DataModel
    {
        public DataModel(int userId, string activityName, long timeStamp, string xacc, string yacc, string zacc)
        {
            UserId = userId;
            ActivityName = activityName;
            TimeStamp = timeStamp;
            DateTime = DateTimeOffset.FromUnixTimeMilliseconds((long)(timeStamp/1000000M));
            Xacc = xacc;
            Yacc = yacc;
            Zacc = zacc;
        }

        public int UserId { get; set; }
        public string ActivityName { get; set; }
        public DateTimeOffset DateTime { get; set; }
        public long TimeStamp { get; set; }
        public long DifferenceInNanoSeconds { get; set; }
        public string Xacc { get; set; }
        public string Yacc { get; set; }
        public string Zacc { get; set; }
    }
}
