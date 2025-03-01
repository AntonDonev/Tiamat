using System.ComponentModel.DataAnnotations;

namespace Tiamat.WebApp.Models
{
    public class RegisterUserViewModel
    {
        [Required]
        public string UserName { get; set; }

        [Required]
        [EmailAddress]
        public string Email { get; set; }
    }
}
