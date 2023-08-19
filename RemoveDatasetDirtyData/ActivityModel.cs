using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RemoveDatasetDirtyData
{
    internal class ActivityModel
    {
        public ActivityModel(string activityName)
        { 
            ActivityName = activityName;
            Actions = new List<DataModel>();
        }
         
        public string ActivityName { get; set; }
        public List<DataModel> Actions { get; set; } 
    }
}
