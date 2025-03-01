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
        [Required]
        public Guid UserId { get; set; }

        public User? User { get; set; }

        [Required]
        [MaxLength(50)]
        [Display(Name = "Account Name")]
        public string AccountName { get; set; }

        [Required]
        [Range(0, double.MaxValue, ErrorMessage = "Началният капитал трябва да бъде положителна стойност")]
        [DataType(DataType.Currency)]
        [Display(Name = "Initial Capital")]
        [Column(TypeName = "decimal(18,2)")]
        public decimal InitialCapital { get; set; }

        [Required]
        [Range(0, double.MaxValue, ErrorMessage = "Текущият капитал трябва да бъде положителна стойност")]
        [DataType(DataType.Currency)]
        [Display(Name = "Current Capital")]
        [Column(TypeName = "decimal(18,2)")]
        public decimal CurrentCapital { get; set; }

        [Required]
        [Range(0, double.MaxValue, ErrorMessage = "Най-високият капитал трябва да бъде положителна стойност")]
        [DataType(DataType.Currency)]
        [Display(Name = "Highest Capital")]
        [Column(TypeName = "decimal(18,2)")]
        public decimal HighestCapital { get; set; }

        [Required]
        [Range(0, double.MaxValue, ErrorMessage = "Най-ниският капитал трябва да бъде положителна стойност")]
        [DataType(DataType.Currency)]
        [Display(Name = "Lowest Capital")]
        [Column(TypeName = "decimal(18,2)")]
        public decimal LowestCapital { get; set; }

        [ForeignKey(nameof(AccountSetting))]
        public Guid AccountSettingsId { get; set; }

        public AccountSetting? AccountSetting { get; set; }

        [Required]
        [MaxLength(20)]
        public string Platform { get; set; }

        [Required]
        [MaxLength(50)]
        [Display(Name = "Broker Login")]
        public string BrokerLogin { get; set; }

        [Required]
        [MaxLength(100)]
        [Display(Name = "Broker Password")]
        public string BrokerPassword { get; set; }

        [Required]
        [MaxLength(100)]
        [Display(Name = "Broker Server")]
        public string BrokerServer { get; set; }

        [Required]
        [EnumDataType(typeof(AccountStatus))]
        public AccountStatus Status { get; set; }

        [MaxLength(45)]
        [Display(Name = "Affiliated IP")]
        public string? Affiliated_IP { get; set; }

        [MaxLength(100)]
        [Display(Name = "VPS Name")]
        public string? VPSName { get; set; }

        [MaxLength(100)]
        [DataType(DataType.EmailAddress)]
        [EmailAddress]
        [Display(Name = "Admin Email")]
        public string? AdminEmail { get; set; }

        [Required]
        [DataType(DataType.DateTime)]
        [Display(Name = "Created At")]
        public DateTime CreatedAt { get; set; }

        [DataType(DataType.DateTime)]
        [Display(Name = "Last Updated At")]
        public DateTime? LastUpdatedAt { get; set; }

        public virtual ICollection<Position> AccountPositions { get; set; }
    }
}
