using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tiamat.Models
{
    public class Notification
    {
        public Guid Id { get; set; }
        public string Title { get; set; }
        public string Description { get; set; }
        public int TotalReadCount { get; set; }
        public DateTime DateTime { get; set; }
        public ICollection<NotificationUser> NotificationUsers { get; set; }
        
        [System.ComponentModel.DataAnnotations.Schema.NotMapped]
        public bool IsReadByCurrentUser { get; set; }

        public Notification()
        {
            NotificationUsers = new List<NotificationUser>();
        }
    }
}
