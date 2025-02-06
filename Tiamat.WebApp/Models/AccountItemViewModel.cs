namespace Tiamat.WebApp.Models
{
    public class AccountItemViewModel
    {
        public Guid AccountId { get; set; }
        public string AccountName { get; set; } = string.Empty;
        public decimal InitialCapital { get; set; }
        public decimal HighestCapital { get; set; }
        public decimal LowestCapital { get; set; }
        public string Platform { get; set; } = string.Empty;
        public string Status { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; }

        public Guid? AccountSettingId { get; set; }
        public string? AccountSettingName { get; set; }
    }
}
