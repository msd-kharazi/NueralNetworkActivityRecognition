using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RemoveDatasetDirtyData
{
    internal class UserActivityModel
    {
        public UserActivityModel(int userId)
        {
            UserId = userId;
            Activities = new List<ActivityModel>();
        }

        public int UserId { get; set; }        
        public List<ActivityModel> Activities { get; set; } 
    }
}
