namespace Tiamat.WebApp.Models
{
    public class AccountSettingItemViewModel
    {
        public Guid AccountSettingId { get; set; }
        public string SettingName { get; set; }
        public int MaxRiskPerTrade { get; set; }
        public int UntradablePeriodMinutes { get; set; }
        public double RiskReward { get; set; }
    }
}
