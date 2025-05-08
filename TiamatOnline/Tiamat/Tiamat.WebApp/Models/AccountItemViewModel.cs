using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Tiamat.Models;

namespace Tiamat.WebApp.Models
{
    public class AccountItemViewModel
    {
        [Key]
        public Guid AccountId { get; set; }
        [Required(ErrorMessage = "Името на акаунта е задължително")]
        [MaxLength(50, ErrorMessage = "Името на акаунта не може да надвишава 50 символа")]
        public string AccountName { get; set; } = string.Empty;
        [Required(ErrorMessage = "Началният капитал е задължителен")]
        [Range(0, double.MaxValue, ErrorMessage = "Началният капитал трябва да бъде положителна стойност")]
        [DataType(DataType.Currency)]
        [Column(TypeName = "decimal(18,2)")]
        public decimal InitialCapital { get; set; }
        [DataType(DataType.Currency)]
        public decimal CurrentCapital { get; set; } 
        [DataType(DataType.Currency)]
        public decimal HighestCapital { get; set; }
        [DataType(DataType.Currency)]
        public decimal LowestCapital { get; set; }
        [Required(ErrorMessage = "Платформата е задължителна")]
        [MaxLength(20, ErrorMessage = "Платформата не може да надвишава 20 символа")]
        public string Platform { get; set; } = string.Empty;
        [Required(ErrorMessage = "Статусът е задължителен")]
        public string Status { get; set; } = string.Empty;
        [Required(ErrorMessage = "Датата на създаване е задължителна")]
        [DataType(DataType.DateTime)]
        public DateTime CreatedAt { get; set; }

        public Guid? AccountSettingId { get; set; }
        public string? AccountSettingName { get; set; }
    }
}
