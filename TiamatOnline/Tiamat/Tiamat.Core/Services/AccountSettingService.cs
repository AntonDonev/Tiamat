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

        public async Task<IEnumerable<AccountSetting>> GetAllSettingsAsync()
        {
            return await _context.AccountSettings
                .Include(s => s.User)
                .Include(s => s.Accounts)
                .ToListAsync();
        }

        public async Task<AccountSetting> GetSettingByIdAsync(Guid id)
        {
            return await _context.AccountSettings
                .Include(s => s.User)
                .Include(s => s.Accounts)
                .FirstOrDefaultAsync(s => s.AccountSettingId == id);
        }

        public async Task<IEnumerable<AccountSetting>> GetSettingsForUserAsync(Guid userId)
        {
            return await _context.AccountSettings
                .Where(s => s.UserId == userId || s.UserId == null)
                .Include(s => s.Accounts)
                .ToListAsync();
        }

        public async Task CreateSettingAsync(AccountSetting setting)
        {
            await _context.AccountSettings.AddAsync(setting);
            await _context.SaveChangesAsync();
        }

        public async Task UpdateSettingAsync(AccountSetting setting)
        {
            _context.AccountSettings.Update(setting);
            await _context.SaveChangesAsync();
        }

        public async Task DeleteSettingAsync(Guid id)
        {
            var setting = await _context.AccountSettings.FirstOrDefaultAsync(s => s.AccountSettingId == id);
            if (setting != null)
            {
                _context.AccountSettings.Remove(setting);
                await _context.SaveChangesAsync();
            }
        }
    }
}