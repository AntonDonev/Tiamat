using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tiamat.Models
{
    public class Application
    {
        [Key]
        public Guid Id { get; set; }
        [Required]
        public string FullName { get; set; }
        [Required]

        public string Email { get; set; }
        [Required]

        public DateTime DateOfBirth { get; set; }

        public string? PassportNumber { get; set; }
        [Required]

        public string ResidencyCountry { get; set; }
        [Required]

        public decimal? EstimatedNetWorth { get; set; }
        [Required]

        public bool IsPoliticallyExposedPerson { get; set; }

        [Required]

        public bool AccreditedInvestor { get; set; }

        public string Status { get; set; } = "Pending";
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime? UpdatedAt { get; set; }

        public string? AdminNotes { get; set; }
        [ForeignKey(nameof(User))]
        public Guid AdminId { get; set; }
        public User Admin { get; set; }
        public DateTime? ApprovalDate { get; set; }


        [ForeignKey(nameof(User))]
        public Guid IdentityUserId { get; set; }
        public User User { get; set; }
    }

}
