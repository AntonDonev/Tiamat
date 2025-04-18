using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Tiamat.WebApp.Models
{
    public class AccountSettingViewModel
    {
        [Key]
        public Guid AccountSettingId { get; set; }
        [Required]
        [MaxLength(50)]
        public string SettingName { get; set; } = string.Empty;
    }
}
