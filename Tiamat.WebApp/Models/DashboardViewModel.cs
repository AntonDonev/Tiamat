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
            public string Id { get; set; }
            public Guid AccountId { get; set; }
            public string Type { get; set; }

            public string OpenedAtIso { get; set; }
        }
    }

}
