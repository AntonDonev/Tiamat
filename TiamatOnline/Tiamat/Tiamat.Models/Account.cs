using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tiamat.Models
{
    public enum AccountStatus
    {
        Pending,
        Active,
        Failed,
        Closed
    }
    public class Account
    {
        public Account()
        {
            AccountPositions = new List<Position>();
        }

        public Account(User user, string accountName, decimal initialCapital, AccountSetting accountSetting, string platform, string brokerLogin, string brokerPassword, string brokerServer, DateTime createdAt)
        {
            Id = Guid.NewGuid();
            UserId = user.Id;
            User = user;
            AccountName = accountName;
            InitialCapital = initialCapital;
            HighestCapital = initialCapital;
            LowestCapital = initialCapital;
            CurrentCapital = initialCapital;
            AccountSettingsId = accountSetting.AccountSettingId;
            AccountSetting = accountSetting;
            Status = AccountStatus.Pending;
            CreatedAt = DateTime.UtcNow;
            Platform = platform;
            BrokerLogin = brokerLogin;
            BrokerPassword = brokerPassword;
            BrokerServer = brokerServer;
            CreatedAt = createdAt;
            AccountPositions = new List<Position>();
            Affiliated_IP = null;
        }

        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public Guid Id { get; set; }

        [ForeignKey(nameof(User))]
        [Required(ErrorMessage = "Потребителският идентификатор е задължителен")]
        public Guid UserId { get; set; }

        public User? User { get; set; }

        [Required(ErrorMessage = "Името на акаунта е задължително")]
        [MaxLength(50, ErrorMessage = "Името на акаунта не може да надвишава 50 символа")]
        [Display(Name = "Име на акаунт")]
        public string AccountName { get; set; }

        [Required(ErrorMessage = "Началният капитал е задължителен")]
        [Range(0, double.MaxValue, ErrorMessage = "Началният капитал трябва да бъде положителна стойност")]
        [DataType(DataType.Currency)]
        [Display(Name = "Начален капитал")]
        [Column(TypeName = "decimal(18,2)")]
        public decimal InitialCapital { get; set; }
        public decimal CurrentCapital { get; set; }
        public decimal HighestCapital { get; set; }
        public decimal LowestCapital { get; set; }

        [ForeignKey(nameof(AccountSetting))]
        [Required(ErrorMessage = "Настройките на акаунта са задължителни")]
        [Display(Name = "Настройки на акаунта")]
        public Guid AccountSettingsId { get; set; }

        public AccountSetting? AccountSetting { get; set; }

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

        [MaxLength(45, ErrorMessage = "Свързаният IP не може да надвишава 45 символа")]
        [Display(Name = "Свързан IP")]
        public string? Affiliated_IP { get; set; }

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

        public virtual ICollection<Position> AccountPositions { get; set; }
    }
}