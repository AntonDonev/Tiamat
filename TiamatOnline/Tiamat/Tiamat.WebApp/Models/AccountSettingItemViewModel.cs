using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Tiamat.WebApp.Models
{
    public class AccountSettingItemViewModel
    {
        [Key]
        public Guid AccountSettingId { get; set; }
        [Required]
        [MaxLength(50)]
        public string SettingName { get; set; }
        [Required]
        [Range(1, 100, ErrorMessage = "Максималният риск за сделка трябва да бъде между 1 и 100 процента")]
        public int MaxRiskPerTrade { get; set; }
        [Required]
        [Range(0, int.MaxValue, ErrorMessage = "Периодът, в който не може да се търгува, трябва да бъде положителна стойност")]
        public int UntradablePeriodMinutes { get; set; }
        public double RiskReward { get; set; }
    }
}
