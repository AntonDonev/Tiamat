namespace Tiamat.WebApp.Models
{
    public class AccountSettingCenterViewModel
    {
        public string SettingNameFilter { get; set; }

        public List<AccountSettingItemViewModel> Settings { get; set; } = new List<AccountSettingItemViewModel>();
    }
}
