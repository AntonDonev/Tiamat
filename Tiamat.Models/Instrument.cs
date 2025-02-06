using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tiamat.Models
{
    public class Instrument
    {
        public Instrument() { }
        public Instrument(string symbol)
        {
            Symbol = symbol;
            PositionInstruments = new List<PositionInstrument>();
        }

        [Key]
        public int Id { get; set; } 

        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; } 

        public ICollection<PositionInstrument> PositionInstruments { get; set; }
    }
}
