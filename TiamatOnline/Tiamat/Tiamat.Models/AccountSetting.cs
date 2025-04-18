using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tiamat.Models
{
    public class AccountSetting
    {
        public AccountSetting(string settingName, int maxRiskPerTrade, int untradablePeriodMinutes, User? user)
        {
            AccountSettingId = Guid.NewGuid();
            SettingName = settingName;
            MaxRiskPerTrade = maxRiskPerTrade;
            UntradablePeriodMinutes = untradablePeriodMinutes;
            UserId = user?.Id;
            User = user;
            Accounts = new List<Account>();
        }

        public AccountSetting()
        {
            Accounts = new List<Account>();
        }

        [Key]
        public Guid AccountSettingId { get; set; }

        [Required]
        [MaxLength(50)]
        [Display(Name = "Име на настройка")]
        public string SettingName { get; set; }

        [Required]
        [Range(1, 100, ErrorMessage = "Максималният риск за сделка трябва да бъде между 1 и 100 процента")]
        [Display(Name = "Максимален риск при сделка (%)")]
        public int MaxRiskPerTrade { get; set; }

        [Required]
        [Range(0, int.MaxValue, ErrorMessage = "Периодът, в който не може да се търгува, трябва да бъде положителна стойност")]
        [Display(Name = "Невалиден период (минути)")]
        public int UntradablePeriodMinutes { get; set; }

        [ForeignKey(nameof(User))]
        public Guid? UserId { get; set; }
        public User? User { get; set; }
        public virtual ICollection<Account> Accounts { get; set; }
    }
}