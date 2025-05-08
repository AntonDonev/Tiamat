using System.ComponentModel.DataAnnotations;
using Tiamat.Models;

namespace Tiamat.WebApp.Models
{
    public class ViewAccountViewModel
    {
        [Key]
        public Guid AccountId { get; set; }

        [Required(ErrorMessage = "Името на акаунта е задължително")]
        [MaxLength(50, ErrorMessage = "Името на акаунта не може да надвишава 50 символа")]
        [Display(Name = "Име на акаунт")]
        public string AccountName { get; set; }
        
        [Display(Name = "Настройки на акаунта")]
        public Guid? AccountSettingsId { get; set; } 

        [Required(ErrorMessage = "Началният капитал е задължителен")]
        [Range(0, double.MaxValue, ErrorMessage = "Началният капитал трябва да бъде положителна стойност")]
        [DataType(DataType.Currency)]
        [Display(Name = "Начален капитал")]
        public decimal InitialCapital { get; set; }
        
        [DataType(DataType.Currency)]
        [Display(Name = "Текущ капитал")]
        public decimal CurrentCapital { get; set; }
        
        [DataType(DataType.Currency)]
        [Display(Name = "Най-висок капитал")]
        public decimal HighestCapital { get; set; }
        
        [DataType(DataType.Currency)]
        [Display(Name = "Най-нисък капитал")]
        public decimal LowestCapital { get; set; }
        
        [Required(ErrorMessage = "Платформата е задължителна")]
        [MaxLength(20, ErrorMessage = "Платформата не може да надвишава 20 символа")]
        [Display(Name = "Платформа")]
        public string Platform { get; set; }
        
        [Required(ErrorMessage = "Брокерският логин е задължителен")]
        [MaxLength(50, ErrorMessage = "Брокерският логин не може да надвишава 50 символа")]
        [Display(Name = "Брокерски логин")]
        public string BrokerLogin { get; set; }
        
        [Required(ErrorMessage = "Брокерската парола е задължителна")]
        [MaxLength(100, ErrorMessage = "Брокерската парола не може да надвишава 100 символа")]
        [Display(Name = "Брокерска парола")]
        public string BrokerPassword { get; set; }
        
        [Required(ErrorMessage = "Брокерският сървър е задължителен")]
        [MaxLength(100, ErrorMessage = "Брокерският сървър не може да надвишава 100 символа")]
        [Display(Name = "Брокерски сървър")]
        public string BrokerServer { get; set; }
        
        [Required(ErrorMessage = "Статусът е задължителен")]
        [EnumDataType(typeof(AccountStatus), ErrorMessage = "Невалиден статус на акаунта")]
        [Display(Name = "Статус")]
        public AccountStatus Status { get; set; }
        
        [MaxLength(100, ErrorMessage = "Името на VPS не може да надвишава 100 символа")]
        [Display(Name = "Име на VPS")]
        public string? VPSName { get; set; }
        
        [MaxLength(100, ErrorMessage = "Имейлът на администратора не може да надвишава 100 символа")]
        [DataType(DataType.EmailAddress)]
        [EmailAddress(ErrorMessage = "Невалиден имейл адрес")]
        [Display(Name = "Имейл на администратора")]
        public string? AdminEmail { get; set; }
        
        [Required(ErrorMessage = "Датата на създаване е задължителна")]
        [DataType(DataType.DateTime)]
        [Display(Name = "Дата на създаване")]
        public DateTime CreatedAt { get; set; }
        
        [DataType(DataType.DateTime)]
        [Display(Name = "Последно обновяване")]
        public DateTime? LastUpdatedAt { get; set; }

        // HWID removed from ViewModel as per requirements

        [DataType(DataType.Date)]
        [Display(Name = "От дата")]
        public DateTime? StartDate { get; set; }
        
        [DataType(DataType.Date)]
        [Display(Name = "До дата")]
        public DateTime? EndDate { get; set; }
        
        [Display(Name = "Тип на сделката")]
        public string? TypeFilter { get; set; }
        
        [Display(Name = "Печалба/Загуба")]
        public string? ResultFilter { get; set; }

        public List<PositionViewModel> Positions { get; set; } = new List<PositionViewModel>();
    }

    public class PositionViewModel
    {
        [Key]
        public string PositionId { get; set; }
        
        [Required]
        [Display(Name = "Символ")]
        public string Symbol { get; set; }
        
        [Required]
        [Display(Name = "Тип")]
        public string Type { get; set; }
        
        [Required]
        [Display(Name = "Размер")]
        public decimal Size { get; set; }
        
        [Required]
        [Display(Name = "Риск")]
        public decimal Risk { get; set; }
        
        [Display(Name = "Резултат")]
        public decimal? Result { get; set; }
        
        [Required]
        [DataType(DataType.DateTime)]
        [Display(Name = "Дата на отваряне")]
        public DateTime OpenedAt { get; set; }
        
        [DataType(DataType.DateTime)]
        [Display(Name = "Дата на затваряне")]
        public DateTime? ClosedAt { get; set; }
    }
}
