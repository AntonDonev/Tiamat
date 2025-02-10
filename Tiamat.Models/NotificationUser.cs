using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tiamat.Models
{
    public class NotificationUser
    {
        [Key]
        public Guid Id { get; set; }

        [ForeignKey(nameof(Notification))]

        public Guid NotificationId { get; set; }
        public Notification Notification { get; set; }

        [ForeignKey(nameof(User))]

        public Guid UserId { get; set; }
        public User User { get; set; }

        [Required]
        public bool IsRead { get; set; }
        public DateTime? ReadAt { get; set; }
    }
}
