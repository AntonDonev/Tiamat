using System.ComponentModel.DataAnnotations;

namespace Tiamat.WebApp.Models
{
    public class AccountSettingAddViewModel
    {
        [Required(ErrorMessage = "Името на настройката е задължително поле!")]
        [MaxLength(50)]
        public string SettingName { get; set; }
        [Required(ErrorMessage = "Максималният риск е задължително поле!")]
        [Range(1, 100, ErrorMessage = "Максималният риск за сделка трябва да бъде между 1 и 100 процента")]
        public int MaxRiskPerTrade { get; set; }
        [Required(ErrorMessage = "Невалидният период е задължително поле!")]
        [Range(0, int.MaxValue, ErrorMessage = "Периодът, в който не може да се търгува, трябва да бъде положителна стойност")]
        public int UntradablePeriodMinutes { get; set; }
    }
}
