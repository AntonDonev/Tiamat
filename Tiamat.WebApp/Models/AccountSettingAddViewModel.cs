using System.ComponentModel.DataAnnotations;

namespace Tiamat.WebApp.Models
{
    public class AccountSettingAddViewModel
    {
        [Required]
        public string SettingName { get; set; }
        [Required]
        public int MaxRiskPerTrade { get; set; }
        [Required]
        public int UntradablePeriodMinutes { get; set; }
    }
}
