using System.ComponentModel.DataAnnotations;

namespace Tiamat.WebApp.Models
{
    public class AccountCenterViewModel
    {
        public string? PlatformFilter { get; set; }
        public string? StatusFilter { get; set; }
        public string? AccountSettingFilter { get; set; }

        public List<AccountItemViewModel> Accounts { get; set; } = new List<AccountItemViewModel>();

        public List<AccountSettingViewModel> AccountSettings { get; set; } = new List<AccountSettingViewModel>();
    }
}
