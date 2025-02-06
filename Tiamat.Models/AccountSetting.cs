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
            SettingName = settingName;
            MaxRiskPerTrade = maxRiskPerTrade;
            UntradablePeriodMinutes = untradablePeriodMinutes;
            UserId = user.Id;
            User = user;
        }
        public AccountSetting() { }
        [Key]
        public Guid AccountSettingId { get; set; }

        [Required]
        [MaxLength(50)]
        public string SettingName { get; set; } 

        [Required]
        public int MaxRiskPerTrade { get; set; } 

        [Required]
        public int UntradablePeriodMinutes { get; set; }


        [ForeignKey(nameof(User))]
        public Guid? UserId { get; set; }

        public User? User { get; set; }

        public ICollection<Account> Accounts { get; set; }

    }
}
