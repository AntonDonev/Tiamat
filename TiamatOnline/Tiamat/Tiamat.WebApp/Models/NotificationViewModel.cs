using System.ComponentModel.DataAnnotations;

namespace Tiamat.WebApp.Models
{
    public class NotificationViewModel
    {
        [Required]
        public string Title { get; set; }
        [Required]
        public string Description { get; set; }
        [Required]
        public string Targets { get; set; }
    }
}
