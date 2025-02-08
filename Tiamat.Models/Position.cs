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
            AccountPositions = new List<AccountPosition>();
        }
        public Position(string symbol, Account account, decimal size, decimal risk, decimal result, DateTime openedAt, DateTime? closedAt)
        {
            Symbol = symbol;
            Account = account;
            AccountId = account.Id;
            Size = size;
            Risk = risk;
            Result = result;
            OpenedAt = openedAt;
            ClosedAt = closedAt;
            AccountPositions = new List<AccountPosition>();
        }

        [Key]
        public Guid Id { get; set; }

        [Required]
        public string Symbol { get; set; }

        [ForeignKey(nameof(Account))]
        public Guid AccountId { get; set; }
        public Account Account { get; set; } 


        [Required]
        public decimal Size { get; set; } 

        [Required]
        public decimal Risk { get; set; } 

        public decimal Result { get; set; } 

        [Required]
        public DateTime OpenedAt { get; set; }

        public DateTime? ClosedAt { get; set; }

        public virtual ICollection<AccountPosition> AccountPositions { get; set; }

    }
}
