using System.ComponentModel.DataAnnotations;

namespace Tiamat.WebApp.Models.Account1
{
    public class DenyAccountViewModel
    {
        [Required(ErrorMessage = "ID на акаунта е задължителен")]
        public Guid Id { get; set; }

        [Required(ErrorMessage = "Заглавието е задължително")]
        public string Title { get; set; }

        [Required(ErrorMessage = "Съобщението е задължително")]
        [StringLength(1000, MinimumLength = 10, ErrorMessage = "Съобщението трябва да бъде между 10 и 1000 символа")]
        public string Message { get; set; }

        public bool UseDefaultDenyMessage { get; set; }
    }
}