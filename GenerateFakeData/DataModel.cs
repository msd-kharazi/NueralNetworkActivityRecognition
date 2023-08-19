using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GenerateFakeData
{
    internal class DataModel
    {
        public DataModel(int userId, string activityName, long timeStamp, string xacc, string yacc, string zacc)
        {
            UserId = userId;
            ActivityName = activityName;
            TimeStamp = timeStamp;
            Xacc = xacc;
            Yacc = yacc;
            Zacc = zacc;
        }

        public int UserId { get; set; }
        public string ActivityName { get; set; }
        public long TimeStamp { get; set; }
        public string Xacc { get; set; }
        public string Yacc { get; set; }
        public string Zacc { get; set; }
    }
}
