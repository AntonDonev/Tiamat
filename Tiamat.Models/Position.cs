using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tiamat.Models
{
    public class Position
    {
        public Position()
        { 
        }
        public Position(string symbol, Account account,string type, decimal size, decimal risk, decimal? result, DateTime openedAt, DateTime? closedAt)
        {
            Symbol = symbol;
            Account = account;
            AccountId = account.Id;
            Type = type;
            Size = size;
            Risk = risk;
            Result = result;
            OpenedAt = openedAt;
            ClosedAt = closedAt;
        }

        [Key]
        public string Id { get; set; }

        [Required]
        public string Symbol { get; set; }

        [Required]
        public string Type { get; set; }

        [ForeignKey(nameof(Account))]
        public Guid AccountId { get; set; }
        public Account Account { get; set; } 

        [Required]
        public decimal Size { get; set; } 

        [Required]
        public decimal Risk { get; set; } 

        public decimal? Result { get; set; } 

        [Required]
        public DateTime OpenedAt { get; set; }

        public DateTime? ClosedAt { get; set; }


    }
}
