using System.ComponentModel.DataAnnotations;

namespace Tiamat.WebApp.Models.Account1
{
    public class ApproveAccountViewModel
    {
        [Required(ErrorMessage = "ID на акаунта е задължителен")]
        public Guid Id { get; set; }

        [Required(ErrorMessage = "Заглавието е задължително")]
        public string Title { get; set; }

        [Required(ErrorMessage = "Име на VPS е задължително")]
        [StringLength(100, MinimumLength = 2, ErrorMessage = "Името на VPS трябва да бъде между 2 и 100 символа")]
        public string VPSName { get; set; }

        [Required(ErrorMessage = "Свързаният HWID е задължителен")]
        [StringLength(100, MinimumLength = 3, ErrorMessage = "HWID трябва да бъде между 3 и 100 символа")]
        public string AffiliatedHWID { get; set; }

        [Required(ErrorMessage = "Съобщението е задължително")]
        [StringLength(1000, MinimumLength = 10, ErrorMessage = "Съобщението трябва да бъде между 10 и 1000 символа")]
        public string Message { get; set; }

        public bool UseDefaultMessage { get; set; }
    }
}