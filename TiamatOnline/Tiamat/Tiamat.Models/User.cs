using Microsoft.AspNetCore.Identity;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;

namespace Tiamat.Models
{
    public class User : IdentityUser<Guid>
    {
        public User()
        {
            AccountSettings = new List<AccountSetting>();
            Accounts = new List<Account>();
            NotificationUsers = new List<NotificationUser>(); 
        }

        public ICollection<AccountSetting> AccountSettings { get; set; }
        public ICollection<Account> Accounts { get; set; }

        public ICollection<NotificationUser> NotificationUsers { get; set; }
    }
}
