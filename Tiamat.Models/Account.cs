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
            AccountPositions = new List<AccountPosition>();
        }

        public Account(User user, string accountName, decimal initialCapital, AccountSetting accountSetting, string platform, string brokerLogin, string brokerPassword, string brokerServer, DateTime createdAt)
        {
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
            AccountPositions = new List<AccountPosition>();

        }

        [Key]
        public Guid Id { get; set; }

        [ForeignKey(nameof(User))]
        public Guid UserId { get; set; }
        public User? User { get; set; }

        [Required]
        [MaxLength(50)]
        public string AccountName { get; set; }

        [Required]
        public decimal InitialCapital { get; set; }

        [Required]
        public decimal CurrentCapital { get; set; }

        [Required]
        public decimal HighestCapital { get; set; }

        [Required]
        public decimal LowestCapital { get; set; }

        [ForeignKey(nameof(AccountSetting))]
        public Guid AccountSettingsId { get; set; }
        public AccountSetting? AccountSetting { get; set; }


        public virtual ICollection<AccountPosition> AccountPositions { get; set; }

        [MaxLength(20)]
        public string Platform { get; set; }

        [MaxLength(50)]
        public string BrokerLogin { get; set; }

        [MaxLength(100)]
        public string BrokerPassword { get; set; }

        [MaxLength(100)]
        public string BrokerServer { get; set; }

        public AccountStatus Status { get; set; }

        [MaxLength(100)]
        public string? VPSName { get; set; }

        public string? AdminEmail { get; set; }

        [Required]
        public DateTime CreatedAt { get; set; }

        public DateTime? LastUpdatedAt { get; set; }
    }
}

