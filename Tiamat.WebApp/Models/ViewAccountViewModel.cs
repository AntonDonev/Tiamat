using Tiamat.Models;

namespace Tiamat.WebApp.Models
{
    public class ViewAccountViewModel
    {
        public Guid AccountId { get; set; }

        public string AccountName { get; set; }
        public Guid? AccountSettingsId { get; set; } 

        public decimal InitialCapital { get; set; }
        public decimal CurrentCapital { get; set; }
        public decimal HighestCapital { get; set; }
        public decimal LowestCapital { get; set; }
        public string Platform { get; set; }
        public string BrokerLogin { get; set; }
        public string BrokerPassword { get; set; }
        public string BrokerServer { get; set; }
        public AccountStatus Status { get; set; }
        public string VPSName { get; set; }
        public string AdminEmail { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime? LastUpdatedAt { get; set; }

        public List<PositionViewModel> Positions { get; set; } = new List<PositionViewModel>();
    }

    public class PositionViewModel
    {
        public Guid PositionId { get; set; }
        public string Symbol { get; set; }
        public decimal Size { get; set; }
        public decimal Risk { get; set; }
        public decimal? Result { get; set; }
        public DateTime OpenedAt { get; set; }
        public DateTime? ClosedAt { get; set; }
    }
}
