using System.ComponentModel.DataAnnotations;

namespace Tiamat.WebApp.Models
{
    public class NotificationViewModel
    {
        [Required(ErrorMessage = "Заглавието е задължително")]
        [StringLength(100, MinimumLength = 3, ErrorMessage = "Заглавието трябва да бъде между 3 и 100 символа")]
        public string Title { get; set; }

        [Required(ErrorMessage = "Описанието е задължително")]
        [StringLength(500, MinimumLength = 10, ErrorMessage = "Описанието трябва да бъде между 10 и 500 символа")]
        public string Description { get; set; }

        [Required(ErrorMessage = "Целевата група е задължителна")]
        public string Targets { get; set; }
    }
}
