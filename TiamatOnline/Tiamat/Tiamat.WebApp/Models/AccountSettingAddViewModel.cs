using System.ComponentModel.DataAnnotations;

namespace Tiamat.WebApp.Models
{
    public class AccountSettingAddViewModel
    {
        [Required]
        [MaxLength(50)]
        public string SettingName { get; set; }
        [Required]
        [Range(1, 100, ErrorMessage = "Максималният риск за сделка трябва да бъде между 1 и 100 процента")]
        public int MaxRiskPerTrade { get; set; }
        [Required]
        [Range(0, int.MaxValue, ErrorMessage = "Периодът, в който не може да се търгува, трябва да бъде положителна стойност")]
        public int UntradablePeriodMinutes { get; set; }
    }
}
