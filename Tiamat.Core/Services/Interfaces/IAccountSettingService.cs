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
        IEnumerable<AccountSetting> GetAllSettings();
        AccountSetting GetSettingById(Guid id);
        void CreateSetting(AccountSetting setting);
        void UpdateSetting(AccountSetting setting);
        void DeleteSetting(Guid id);
        IEnumerable<AccountSetting> GetSettingsForUser(Guid userId);

    }
}
