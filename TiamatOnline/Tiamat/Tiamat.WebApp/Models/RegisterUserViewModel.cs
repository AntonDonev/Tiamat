using System.ComponentModel.DataAnnotations;

namespace Tiamat.WebApp.Models
{
    public class RegisterUserViewModel
    {
        [Required(ErrorMessage = "Потребителското име е задължително")]
        [MaxLength(256, ErrorMessage = "Потребителското име не може да надвишава 256 символа")]
        public string UserName { get; set; }

        [Required(ErrorMessage = "Имейлът е задължителен")]
        [EmailAddress(ErrorMessage = "Невалиден имейл адрес")]
        [MaxLength(256, ErrorMessage = "Имейлът не може да надвишава 256 символа")]
        public string Email { get; set; }
    }
}
