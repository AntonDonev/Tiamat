using System.ComponentModel.DataAnnotations;
using Tiamat.Models;

namespace Tiamat.WebApp.Models
{
    public class DashboardViewModel
    {
        public List<Notification> Notifications { get; set; }

        public List<PositionChartDto> Positions { get; set; }

        public int CurrentPage { get; set; }
        public int TotalPages { get; set; }

        public class PositionChartDto
        {
            [Required]
            public string Id { get; set; }
            
            [Required]
            public Guid AccountId { get; set; }
            
            [Required]
            public string Type { get; set; }

            [Required]
            public string OpenedAtIso { get; set; }
        }
    }

}
