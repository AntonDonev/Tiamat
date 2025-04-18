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
            [MaxLength(45, ErrorMessage = "Свързаният IP не може да надвишава 45 символа")]
            public string FromIp { get; set; }
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
            public string FromIp { get; set; }
        }

        public class StartAccountRequest
        {
            [Required]
            public string AccountId { get; set; }
            [Required]
            [MaxLength(45, ErrorMessage = "Свързаният IP не може да надвишава 45 символа")]
            public string Ip { get; set; }
        }
    }
}
