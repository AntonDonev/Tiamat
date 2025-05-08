using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Tiamat.WebApp.Models
{
    namespace Tiamat.WebApp.ViewModels.Python
    {
        public class OpenConfirmRequest
        {
            [Required]
            [Key]
            public string Id { get; set; }
            [Required]
            public string Symbol { get; set; }
            [Required]
            public string Type { get; set; }
            [Required]
            public decimal Size { get; set; }
            [Required]
            public decimal Risk { get; set; }
            [Required]
            [DataType(DataType.DateTime)]
            public DateTime OpenedAt { get; set; }
            [Required]
            [MaxLength(100, ErrorMessage = "Свързаният HWID не може да надвишава 100 символа")]
            public string FromHwid { get; set; }
        }

        public class ClosedConfirmRequest
        {
            [Required]
            [Key]
            public string Id { get; set; }
            [Required]
            public decimal Profit { get; set; }
            [Required]
            [DataType(DataType.Currency)]
            public decimal CurrentCapital { get; set; }
            [Required]
            [DataType(DataType.DateTime)]
            public DateTime ClosedAt { get; set; }
            public string FromHwid { get; set; }
        }

        public class StartAccountRequest
        {
            [Required]
            public string AccountId { get; set; }
            [Required]
            [MaxLength(100, ErrorMessage = "Свързаният HWID не може да надвишава 100 символа")]
            public string Hwid { get; set; }
        }
    }
}
