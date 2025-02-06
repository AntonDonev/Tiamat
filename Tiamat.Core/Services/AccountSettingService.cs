using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tiamat.Core.Services.Interfaces;
using Tiamat.DataAccess;
using Tiamat.Models;

namespace Tiamat.Core.Services
{
    public class AccountSettingService : IAccountSettingService
    {
        private readonly TiamatDbContext _context;

        public AccountSettingService(TiamatDbContext context)
        {
            _context = context;
        }

        public IEnumerable<AccountSetting> GetAllSettings()
        {
            return _context.AccountSettings
                .Include(s => s.User)
                .Include(s => s.Accounts)
                .ToList();
        }

        public AccountSetting GetSettingById(Guid id)
        {
            return _context.AccountSettings
                .Include(s => s.User)
                .Include(s => s.Accounts)
                .FirstOrDefault(s => s.AccountSettingId == id);
        }

        public IEnumerable<AccountSetting> GetSettingsForUser(Guid userId)
        {
            return _context.AccountSettings
                .Where(s => s.UserId == userId || s.UserId == null) 
                .Include(s => s.Accounts)
                .ToList();
        }

        public void CreateSetting(AccountSetting setting)
        {
            _context.AccountSettings.Add(setting);
            _context.SaveChanges();
        }

        public void UpdateSetting(AccountSetting setting)
        {
            _context.AccountSettings.Update(setting);
            _context.SaveChanges();
        }

        public void DeleteSetting(Guid id)
        {
            var setting = _context.AccountSettings.FirstOrDefault(s => s.AccountSettingId == id);
            if (setting != null)
            {
                _context.AccountSettings.Remove(setting);
                _context.SaveChanges();
            }
        }
    }
}
