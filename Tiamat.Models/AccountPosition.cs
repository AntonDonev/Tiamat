using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tiamat.Models
{
    public class AccountPosition
    {
        [Key]
        public Guid Id { get; set; }

        [ForeignKey(nameof(Account))]
        public Guid AccountId { get; set; }
        public virtual Account Account { get; set; }

        [ForeignKey(nameof(Position))]
        public Guid PositionId { get; set; }
        public virtual Position Position { get; set; }

        [Required]
        public decimal Size { get; set; }

        [Required]
        public decimal Risk { get; set; }

        public decimal Result { get; set; }

        [Required]
        public DateTime OpenedAt { get; set; }

        public DateTime? ClosedAt { get; set; }
    }
}
