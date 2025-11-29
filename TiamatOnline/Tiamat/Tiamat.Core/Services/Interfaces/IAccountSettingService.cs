using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tiamat.Models;

namespace Tiamat.Core.Services.Interfaces
{
    public interface IAccountSettingService
    {
        Task<IEnumerable<AccountSetting>> GetAllSettingsAsync();
        Task<AccountSetting> GetSettingByIdAsync(Guid id);
        Task CreateSettingAsync(AccountSetting setting);
        Task UpdateSettingAsync(AccountSetting setting);
        Task DeleteSettingAsync(Guid id);
        Task<IEnumerable<AccountSetting>> GetSettingsForUserAsync(Guid userId);
        Task<IEnumerable<AccountSetting>> GetFilteredSettingsForUserAsync(Guid userId, string settingNameFilter);
    }
}
